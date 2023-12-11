import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import  Variable
from torch.nn.parameter import Parameter
from qpth.qp import QPFunction

# Helper function to stack zeros
def zeors_hstack_help(vec, n, size_row, size_col):
    combo = vec
    single = Variable(torch.zeros(size_row,size_col).cuda())
    for i in range(n-1):
        combo = torch.hstack((combo,single))
    return combo

# Dataloader 
class Dataset_LeTac(data.Dataset):
    def __init__(self, folders_pv_pair, labels, frames, transform=None):
        self.labels = labels
        self.folders = list(np.array(tuple(folders_pv_pair),dtype=object)[:,0])
        self.pv_pairs = list(np.array(tuple(folders_pv_pair),dtype=object)[:,1])
        self.transform = transform
        self.frames = frames

    def __len__(self):
        return len(self.folders)

    def read_images(self, selected_folder, use_transform):
        image = Image.open(selected_folder[0])
        if use_transform is not None:
            image = use_transform(image)
        return image

    def __getitem__(self, index):
        folder = self.folders[index]
        pv_pair = self.pv_pairs[index]
        x = self.read_images(folder, self.transform)    
        x = (x,tuple(pv_pair))
        y = torch.FloatTensor([self.labels[index]])            
        return x, y

# 2D CNN encoder using ResNet-152 pretrained.
class ResCNNEncoder(nn.Module):
    def __init__(self, hidden1=512, hidden2=512, dropP=0.3, outputDim=300):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(ResCNNEncoder, self).__init__()

        self.hidden1, self.hidden2 = hidden1, hidden2
        self.dropP = dropP

        resnet = models.resnet152(pretrained=True)
        # Delete the last fc layer.
        modules = list(resnet.children())[:-1]      
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1, momentum=0.01)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2, momentum=0.01)
        self.fc3 = nn.Linear(hidden2, outputDim)
        
    def forward(self, x):
        with torch.no_grad():
            x = self.resnet(x[:, :, :, :])  
            x = x.view(x.size(0), -1)            
        x = self.bn1(self.fc1(x))
        x = F.relu(x)
        x = self.bn2(self.fc2(x))
        x = F.relu(x)
        x = F.dropout(x, p=self.dropP, training=self.training)
        x = self.fc3(x)
        return x


# Differentiable MPC layer
class MPClayer(nn.Module):
    def __init__(self,nHidden = 25, eps = 1e-4, nStep = 20, del_t = 1/60):
        super(MPClayer, self).__init__()

        self.Pq = 5
        self.Qv = 200
        self.Qa = 1
        self.nHidden = nHidden
        self.eps = eps
        self.nStep = nStep
        self.del_t = del_t
    
        # A matrix
        self.A_eye = Variable(torch.eye(self.nHidden).cuda())
        self.Af_zero = Variable(torch.zeros(self.nHidden,1).cuda())
        self.Af= Parameter(torch.rand(self.nHidden,1).cuda())
        Ap_right_temp = Variable(torch.from_numpy(np.array([
        [1,     self.del_t],
        [0,     1]
        ])).float().cuda())
        self.Ap_right = torch.hstack((Variable(0*torch.ones(2, self.nHidden).cuda()),Ap_right_temp))

        # B matrix
        Bg = Variable(torch.from_numpy(np.array([
        [0.5*self.del_t*self.del_t],
        [self.del_t]
        ])).float().cuda())
        self.B_zero = Variable(torch.zeros(self.nHidden, 1).cuda())
        self.B0 = torch.vstack((self.B_zero,Bg))

        # Weights
        self.Lq = Parameter(torch.tril(torch.rand(self.nHidden, self.nHidden).cuda()))
        self.R0 = self.Qa*Variable(torch.eye(1).cuda())
        self.Q0_right_down = Variable(torch.from_numpy(np.array([
        [0,     0],
        [0,     self.Qv]
        ])).float().cuda())
        self.Q0_down = Variable(torch.zeros(2,self.nHidden).cuda())
        self.Q0_right = Variable(torch.zeros(self.nHidden,2).cuda())

        # No constraints during training
        self.G = Variable(torch.zeros(self.nStep,self.nStep).cuda())
        self.h = Variable(torch.zeros(self.nStep,1).cuda())


    def forward(self, x, gripper_p, gripper_v):
        nBatch = x.size(0)

        # Single Q in cost function
        Q0 = self.Lq.mm(self.Lq.t()) + self.eps*Variable(torch.eye(self.nHidden)).cuda()
        Q0 = torch.hstack((Q0,self.Q0_right))
        Q0 = torch.vstack((Q0,torch.hstack((self.Q0_down,self.Q0_right_down))))

        # Stacked Q
        Q0_stack = Q0.unsqueeze(0).expand(self.nStep-1, self.nHidden+2, self.nHidden+2)
        Q0_final = self.Pq*Q0.unsqueeze(0).expand(1, self.nHidden+2, self.nHidden+2)
        Q0_stack = torch.vstack((Q0_stack,Q0_final))
        Q_dia =  torch.block_diag(*Q0_stack).cuda()

        # Stacked R
        R0_stack = self.R0.unsqueeze(0).expand(self.nStep, 1, 1)
        R_dia =  torch.block_diag(*R0_stack).cuda()

        # Model computing
        A0 = torch.vstack((torch.hstack((torch.hstack((self.A_eye,self.Af_zero)),self.Af)),self.Ap_right))
        T_ = A0
        temp = A0
        for i in range(self.nStep-1):
            temp = torch.mm(temp,A0)
            T_ = torch.vstack((T_,temp))
        I=Variable(torch.eye(self.nHidden+2).cuda())
        row_single = zeors_hstack_help(I, self.nStep, self.nHidden+2, self.nHidden+2)
        AN_ = row_single
        for i in range(self.nStep-1):
            AN = I
            row_single = I
            for j in range(i+1):
                AN = torch.mm(A0,AN)
                row_single = torch.hstack((AN,row_single))
            row_single = zeors_hstack_help(row_single, self.nStep-i-1, self.nHidden+2, self.nHidden+2)
            AN_=torch.vstack((AN_, row_single))
        B0_stack = self.B0.unsqueeze(0).expand(self.nStep, self.nHidden+2, 1)
        B_dia =  torch.block_diag(*B0_stack)
        S_ = torch.mm(AN_,B_dia)
        Q_final = 2*(R_dia+(torch.mm(S_.t(),Q_dia)).mm(S_))+ self.eps*Variable(torch.eye(self.nStep)).cuda()
        Q_batch = Q_final.unsqueeze(0).expand(nBatch, self.nStep, self.nStep)
        p_final = 2*torch.mm(T_.t(),torch.mm(Q_dia,S_))
        p_batch = p_final.unsqueeze(0).expand(nBatch, self.nHidden+2, self.nStep)

        gripper_p = gripper_p.reshape([nBatch,1]).float()
        gripper_v = gripper_v.reshape([nBatch,1]).float()
        gripper_state = torch.hstack((gripper_p,gripper_v))
        
        x = torch.hstack((x,gripper_state))
        x = x.reshape([nBatch,1,self.nHidden+2])
        p_x0_batch = torch.bmm(x,p_batch)
        e = Variable(torch.Tensor())
        G = self.G.unsqueeze(0).expand(nBatch, self.nStep, self.nStep)
        h = self.h.unsqueeze(0).expand(nBatch, self.nStep, 1)
        p_x0_batch = p_x0_batch.reshape([nBatch,self.nStep])
        h = h.reshape([nBatch, self.nStep])

        u = QPFunction(verbose=-1)(Q_batch, p_x0_batch, G, h, e, e)

        S_batch = S_.unsqueeze(0).expand(nBatch, self.nStep*(self.nHidden+2), self.nStep)
        T_batch = T_.unsqueeze(0).expand(nBatch, self.nStep*(self.nHidden+2), self.nHidden+2)
        x_predict = torch.bmm(S_batch,u.reshape(nBatch,self.nStep,1)) + torch.bmm(T_batch,x.reshape(nBatch,self.nHidden+2,1))
        embb_output = Variable(torch.zeros(1,self.nHidden).cuda())
        state_output = Variable(torch.eye(1).cuda())
        output_single = torch.hstack((embb_output,state_output))
        output_single = torch.hstack((output_single,torch.zeros(1,1).cuda()))
        output_stack = output_single.unsqueeze(0).expand(self.nStep, 1, self.nHidden+2)
        output_dia =  torch.block_diag(*output_stack).cuda()
        output_batch = output_dia.unsqueeze(0).expand(nBatch, 1*self.nStep, self.nStep*(self.nHidden+2))
        posi_predict = torch.bmm(output_batch,x_predict).resize(nBatch,self.nStep)
        x = posi_predict
        return x




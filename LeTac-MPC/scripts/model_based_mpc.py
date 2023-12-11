import numpy as np
from scipy import sparse
import time
import sys, tty,termios
import rospy
from wsg_50_common.msg import Cmd
from wsg_50_common.msg import Status
from std_msgs.msg import Float32

import osqp

def vstack_help(vec, n):
    combo = vec.reshape(vec.size,1)
    single  = vec.reshape(vec.size,1)
    for i in range(n-1):
        combo = np.vstack((combo,single))
    return combo

def zeors_hstack_help(vec, n, size_row, size_col):
    combo = vec
    single  = sparse.csc_matrix((size_row, size_col), dtype=np.int8)
    for i in range(n-1):
        combo = sparse.hstack((combo,single))
    return combo

def zeors_hstack_help_inverse(vec, n, size_row, size_col):
    end = vec
    single  = sparse.csc_matrix((size_row, size_col), dtype=np.int8)
    combo = single
    for i in range(n-2):
        combo = sparse.hstack((combo,single))
    combo = sparse.hstack((combo,end))
    return combo

def getCS_(C,S_):
    C_ = sparse.block_diag([sparse.kron(sparse.eye(N), C)])
    return C_*S_

def getCT_(C,T_):
    C_ = sparse.block_diag([sparse.kron(sparse.eye(N), C)])
    return C_*T_

def b_CT_x0(b_,CT_,x0):
    return b_ - CT_*x0

global gripper_posi_
global gripper_ini_flag_
global dis_sum_
global contact_area_

contact_area_ = 0
dis_sum_= 0
gripper_posi_ = 0
gripper_ini_flag_ = False

def contact_area_cb(msg):
    global contact_area_
    contact_area_ = msg.data

def dis_sum_cb(msg):
    global dis_sum_
    dis_sum_ = msg.data

def gripper_state_cb(data):
    global gripper_posi_
    global gripper_ini_flag_
    gripper_posi_ = data.width
    gripper_ini_flag_ = True

if __name__ == "__main__":
    rospy.init_node('model_based_mpc_node', anonymous=True)

    gripper_posi_pub = rospy.Publisher('/wsg/goal_position', Cmd, queue_size = 1)

    rospy.Subscriber("/wsg/status", Status, gripper_state_cb)
    rospy.Subscriber('/tactile_state/marker_dis_sum', Float32,dis_sum_cb)
    rospy.Subscriber('/tactile_state/contact_area', Float32,contact_area_cb)

    old_attr = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    # Parameters initialization
    frequency = 60
    init_posi = 70
    N = 15 # horizon
    q_c = 36
    q_v = 1
    q_d = 2
    q_a = 2
    p=5
    c_ref = 1000
    k_c= 36000
    acc_max = 30
    vel_max = 10
    dim = 4

    del_t=1/frequency 
    gripper_cmd = Cmd()
    gripper_cmd.pos = init_posi
    rate = rospy.Rate(frequency)

    try:
        while not gripper_ini_flag_:
           print('Wait for initializing the gripper.')

        while (sys.stdin.read(1) != 'l'):
            print('Wait for starting! Press l to start')
            time.sleep(0.1)
        
        # state and control Initialization
        x_state = np.array([0.,0.,0.,0.])
        u0 =np.array([[0.]])

        # reference to track
        r = np.array([c_ref,0,0,0]) 
        r_ = vstack_help(r,N)

        # model
        Ad = sparse.csc_matrix([
        [1,   0,    0,  k_c*del_t],
        [0,   1,    0,          0],
        [0,   0,    1,     -del_t],
        [0,   0,    0,          1]
        ])

        Bd = sparse.csc_matrix([
        [0],
        [0],
        [-0.5*del_t*del_t],
        [del_t]
        ])

        # weights
        Q = sparse.csc_matrix([
        [q_c, q_c*q_d, 0, 0],
        [q_c*q_d, q_c*(q_d**2), 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, q_v]
        ])
        R =q_a*sparse.eye(1)
        QN = p*Q
        Q_ = sparse.block_diag([sparse.kron(sparse.eye(N-1), Q), QN], format='csc')
        R_ = sparse.block_diag([sparse.kron(sparse.eye(N), R)], format='csc')

        # T initialization
        T_ = Ad
        temp = Ad
        for i in range(N-1):
            temp = temp.dot(Ad)
            T_ = sparse.vstack([T_,temp])

        I = sparse.eye(dim)
        row_single = zeors_hstack_help(I, N, dim, dim)
        AN_ = row_single
        for i in range(N-1):
            AN = I
            row_single = I
            for j in range(i+1):
                AN = Ad.dot(AN)
                row_single = sparse.hstack([AN,row_single])
            row_single = zeors_hstack_help(row_single, N-i-1, dim, dim)
            AN_=sparse.vstack([AN_, row_single])

        Bd_ = sparse.block_diag([sparse.kron(sparse.eye(N), Bd)])
        S_ = AN_*Bd_ 

        # vel and acc constraints
        max_con_b = (np.array([vel_max])).reshape(1,1)
        min_con_b = (np.array([-vel_max])).reshape(1,1)
        u_max = acc_max*np.ones(1*N)
        u_max = u_max.reshape(1*N,1)

        max_con_b_ = vstack_help(max_con_b,N)
        min_con_b_ = vstack_help(min_con_b,N)

        # vel selct matrix
        C_con = sparse.csc_matrix([
        [0,0,0,1]
        ])

        C_con_T_ = getCT_(C_con,T_)

        # real-time vel bounds
        max_con_b_update = b_CT_x0(max_con_b_,C_con_T_,x_state.reshape(dim,1))
        min_con_b_update = b_CT_x0(min_con_b_,C_con_T_,x_state.reshape(dim,1))

        u_ = np.vstack([u_max,max_con_b_update])
        l_ = np.vstack([u_max*-1,min_con_b_update])

        # select matrix for cost function
        L = sparse.eye(dim)
        L_ = sparse.block_diag([sparse.kron(sparse.eye(N), L)], format='csc')

        # QP setup
        P_=2*(R_+(S_.T)*(L_.T)*Q_*L_*S_)
        q_ = 2*(x_state.reshape(1,dim)*(T_.T)*(L_.T)-r_.T)*Q_*L_*S_
        A_=sparse.vstack([sparse.block_diag([sparse.eye(1*N)], format='csc'),getCS_(C_con,S_) ])

        prob = osqp.OSQP()
        prob.setup(P_, q_.T, A_, l_, u_, warm_start=True, max_iter = 8000)
        initial_flag = False

        while not rospy.is_shutdown():

            if x_state[2] == 0.:
                # state initialization
                x_state = np.array([contact_area_, -dis_sum_,gripper_cmd.pos,x_state[3]])
            else:
                # tactile state update
                # contact area, dis sum, p, v
                x_state = np.array([contact_area_, -dis_sum_,x_state[2],x_state[3]])

            # constraints update
            max_con_b_update = b_CT_x0(max_con_b_,C_con_T_,x_state.reshape(dim,1))
            min_con_b_update = b_CT_x0(min_con_b_,C_con_T_,x_state.reshape(dim,1))
            u_ = np.vstack([u_max,max_con_b_update])
            l_ = np.vstack([u_max*-1,min_con_b_update])

            # QP update
            q_ = 2*(x_state.reshape(1,dim)*(T_.T)*(L_.T)-r_.T)*Q_*L_*S_
            prob.update(q=q_.T, l=l_, u=u_)
            res = prob.solve()
            ctrl = res.x[0:1].copy()

            if ctrl[0] is not None:
                # p, v update
                x_state = Ad.dot(x_state) + Bd.dot(ctrl)
                gripper_cmd.pos = x_state[2]

            gripper_posi_pub.publish(gripper_cmd)
            rate.sleep()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_attr)
    except KeyboardInterrupt:
            print('Interrupted!')




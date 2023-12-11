import numpy as np
from scipy import sparse
import time
import sys, tty,termios
import rospy
from wsg_50_common.msg import Cmd
from wsg_50_common.msg import Status
from std_msgs.msg import Float32

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
    rospy.init_node('model_based_pd_node', anonymous=True)

    gripper_posi_pub = rospy.Publisher('/wsg/goal_position', Cmd, queue_size = 1)

    rospy.Subscriber("/wsg/status", Status, gripper_state_cb)
    rospy.Subscriber('/tactile_state/marker_dis_sum', Float32,dis_sum_cb)
    rospy.Subscriber('/tactile_state/contact_area', Float32,contact_area_cb)

    old_attr = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    # Parameters initialization
    frequency = 60
    init_posi = 70

    q_d = 2
    c_ref = 1000
    k_p= 1/40000
    k_d = 1/6000

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
        
        last_contact_area_ = 0
        while not rospy.is_shutdown():

            gripper_cmd.pos =  gripper_cmd.pos + (contact_area_ - (c_ref+(q_d*dis_sum_)))*k_p + (contact_area_ - last_contact_area_)*k_d
            gripper_posi_pub.publish(gripper_cmd)
            last_contact_area_ = contact_area_

            rate.sleep()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_attr)
    except KeyboardInterrupt:
            print('Interrupted!')




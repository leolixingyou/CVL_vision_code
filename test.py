import os
import time
import math
import numpy as np
import copy
import cv2
import argparse

from detection.det_infer import Predictor
from detection.calibration import Calibration
from segmentation.seg_infer import BaseEngine

import rospy
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseArray, Pose

###
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
###

class Camemra_Node:
    def __init__(self,args):
        rospy.init_node('RTra_node')
        self.args = args
        
        local = os.getcwd()
        print('now is here', local)
        camera_path = [
                    './detection/calibration_data/epiton_cal/f60.txt',
                    './detection/calibration_data/epiton_cal/f120.txt',
                    './detection/calibration_data/epiton_cal/r120.txt'
                    ]
        self.calib = Calibration(camera_path)

        self.get_f60_new_image = False
        self.cur_f60_img = {'img':None, 'header':None}
        self.sub_f60_img = {'img':None, 'header':None}

        rospy.Subscriber('/gmsl_camera/dev/video0/compressed', CompressedImage, self.IMG_f60_callback, queue_size=2)

       ##########################
        self.pub_f60_det = rospy.Publisher('/det_result/f60', Image, queue_size=1)
        self.bridge = CvBridge()
        self.is_save =False
        self.sup = []
        ##########################
         
    def IMG_f60_callback(self,msg):
        self.temp1 = time.time()
        if not self.get_f60_new_image: 
            np_arr = np.frombuffer(msg.data, np.uint8)
            front_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            print('ENCODING is :', round(1/(time.time() - self.temp1),2),' FPS')
            self.temp = time.time()
            front_img = cv2.resize(front_img,(1280,720))
            print('RESIZE is :', round(1/(time.time() - self.temp1),2),' FPS')

            self.temp = time.time()

            self.cur_f60_img['img'] = self.calib.undistort(front_img,'f60')

            print('Distorted is :', round(1/(time.time() - self.temp),2),' FPS')

            self.cur_f60_img['header'] = msg.header
            self.get_f60_new_image = True
            print('rate is :', round(1/(time.time() - self.temp1),2),' FPS')

    def main(self):
        while not rospy.is_shutdown():
            self.t1 = time.time()
            if self.get_f60_new_image:
                self.get_f60_new_image= False



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--det_weight', default="./detection/weights/yolov7x_yellow_edn2end.trt")  
    parser.add_argument('--seg_weight', default="./segmentation/weights/None_384x640_sim_3.trt")  
    # parser.add_argument('--seg_weight', default="./segmentation/weights/hybridnets_c0_384x640_simplified.trt")  

    parser.add_argument("--end2end", default=True, action="store_true",help="use end2end engine")
    parser.add_argument('--anchor', default='./segmentation/anchors/None_anchors_384x640.npy')
    # parser.add_argument('--nc', type=str, default='10', help='Number of detection classes')
    parser.add_argument('--nc', type=str, default='1', help='Number of detection classes')
    
    args = parser.parse_args()

    Camemra_Node = Camemra_Node(args)
    Camemra_Node.main()


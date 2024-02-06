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
    def __init__(self,args,day_night):
        rospy.init_node('Camemra_node')
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
        self.bbox_f60 = PoseArray()

        self.get_f120_new_image = False
        self.cur_f120_img = {'img':None, 'header':None}
        self.sub_f120_img = {'img':None, 'header':None}

        self.get_r120_new_image = False
        self.cur_r120_img = {'img':None, 'header':None}
        self.sub_r120_img = {'img':None, 'header':None}

        ### Seg Model initiolization
        seg_model = args.seg_weight
        seg_input_size = (384, 640)
        anchors = args.anchor if args.anchor else None
        nc = int(args.nc)
        self.seg_pred = BaseEngine(seg_model, seg_input_size, anchors, nc)

        ### Det Model initiolization
        self.det_pred = Predictor(engine_path=args.det_weight , day_night=day_night)
        
        self.pub_od_f60 = rospy.Publisher('/mobinha/perception/camera/bounding_box', PoseArray, queue_size=1)
        
        rospy.Subscriber('/gmsl_camera/dev/video0/compressed', CompressedImage, self.IMG_f60_callback)
        # rospy.Subscriber('/gmsl_camera/dev/video1/compressed', CompressedImage, self.IMG_f120_callback)
        # rospy.Subscriber('/gmsl_camera/dev/video2/compressed', CompressedImage, self.IMG_r120_callback)

       ##########################
        self.pub_f60_det = rospy.Publisher('/det_result/f60', Image, queue_size=1)
        self.pub_f60_seg = rospy.Publisher('/seg_result/f60', Image, queue_size=1)
        self.pub_f120_seg = rospy.Publisher('/seg_result/f120', Image, queue_size=1)
        self.pub_r120_seg = rospy.Publisher('/seg_result/r120', Image, queue_size=1)
        self.bridge = CvBridge()
        self.is_save =False
        self.sup = []
        ##########################
         
    def IMG_f60_callback(self,msg):
        self.temp = time.time()
        if not self.get_f60_new_image:
            np_arr = np.fromstring(msg.data, np.uint8)
            front_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            self.cur_f60_img['img'] = self.calib.undistort(front_img,'f60')
            self.cur_f60_img['header'] = msg.header
            self.get_f60_new_image = True
            # print('rate is :', round(1/(time.time() - self.temp),2),' FPS')

    def IMG_f120_callback(self,msg):
        if not self.get_f120_new_image:
            np_arr = np.fromstring(msg.data, np.uint8)
            front_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            self.cur_f120_img['img'] = self.calib.undistort(front_img,'f120')
            self.cur_f120_img['header'] = msg.header
            self.get_f120_new_image = True

    def IMG_r120_callback(self,msg):
        if not self.get_r120_new_image:
            np_arr = np.fromstring(msg.data, np.uint8)
            front_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            self.cur_r120_img['img'] = self.calib.undistort(front_img,'r120')
            self.cur_r120_img['header'] = msg.header
            self.get_r120_new_image = True

    def pose_set(self,bboxes,flag):
        bbox_f60 = PoseArray()
        for bbox in bboxes:
            pose = Pose()
            pose.position.x = bbox[0]# box class
            pose.position.y = bbox[1]# box area
            pose.position.z = bbox[2]# box score
            pose.orientation.x = bbox[3][0]# box mid x
            pose.orientation.y = bbox[3][1]# box mid y
            pose.orientation.z = bbox[3][2]# box mid y
            pose.orientation.w = bbox[3][3]# box mid y
            bbox_f60.poses.append(pose)
        self.pub_od_f60.publish(bbox_f60)
        

    def det_pubulissher(self,det_img,det_box,flag):
        temp = time.time()
        det_f60_msg = self.bridge.cv2_to_imgmsg(det_img, "bgr8")#color
        self.pose_set(det_box,flag)
        #self.pub_od_f60.publish(self.bbox_f60)
        self.pub_f60_det.publish(det_f60_msg)
        # print('f60 publishing is :', round((time.time() - temp)*1000,2),' ms')
        # print('f60 whole processing is :', round((time.time() - self.t1)*1000,2),' ms')

    def seg_pubulissher(self,seg_img,flag):
        temp = time.time()
        # if flag == 'f60':
        #     seg_img = seg_img *30
        #     seg_f120_msg = self.bridge.cv2_to_imgmsg(np.array(seg_img), "mono8")#gray
        #     self.pub_f60_seg.publish(seg_f120_msg)
        #     print('f60 publishing is :', round((time.time() - temp)*1000,2),' ms')
        #     print('f60 whole processing is :', round((time.time() - self.t2)*1000,2),' ms')

        if flag == 'f120':
            seg_f120_msg = self.bridge.cv2_to_imgmsg(np.array(seg_img), "mono8")#gray
            self.pub_f120_seg.publish(seg_f120_msg)
            # print('f120 publishing is :', round((time.time() - temp)*1000,2),' ms')
            # print('f120 whole processing is :', round((time.time() - self.t2)*1000,2),' ms')

        if flag == 'r120':
            seg_r120_msg = self.bridge.cv2_to_imgmsg(np.array(seg_img), "mono8")#gray
            self.pub_r120_seg.publish(seg_r120_msg)
            # print('r120 publishing is :', round((time.time() - temp)*1000,2),' ms')
            # print('r120 whole processing is :', round((time.time() - self.t3)*1000,2),' ms')
            
    def image_process(self,img,flag):
        try:
            if flag == 'f60' :
                det_img, box_result = self.det_pred.steam_inference(img,conf=0.1, end2end=args.end2end)
                self.det_pubulissher(det_img,box_result,flag)
                
            # if flag == 'f120' :
            #     self.seg_pred.inference(img)
            #     seg_img, t_ms = self.seg_pred.draw_2D(img)
            #     seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)
            #     self.seg_pubulissher(seg_img,flag)
            
            # if flag == 'r120' :
            #     self.seg_pred.inference(img)
            #     seg_img, t_ms = self.seg_pred.draw_2D(img)
            #     seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)
            #     self.seg_pubulissher(seg_img,flag)
        except CvBridgeError as e:
            print(e)

    def main(self):
        while not rospy.is_shutdown():
            self.t1 = time.time()
            
            if self.get_f60_new_image:
                # print('f60 ------')
                orig_im_f60 = copy.copy(self.cur_f60_img['img']) 
                self.sub_f60_img['img'] = self.cur_f60_img['img']
                self.image_process(orig_im_f60,'f60')
                self.get_f60_new_image = False
                
            # self.t2 = time.time()
            # if self.get_f120_new_image:
            #     # print('f120 =====')
            #     orig_im_f120 = copy.copy(self.cur_f120_img['img']) 
            #     self.image_process(orig_im_f120,'f120')
            #     self.get_f120_new_image = False

            # self.t3 = time.time()
            # if self.get_r120_new_image:
            #     # print('r120 !!!!!!')
            #     orig_im_r120 = copy.copy(self.cur_r120_img['img']) 
            #     self.image_process(orig_im_r120,'r120')
            #     self.get_r120_new_image = False
            
            # print('PRO is :', round(1/(time.time() - self.temp),2),' FPS')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--det_weight', default="./detection/weights/yolov7x_yellow_edn2end.trt")  
    
    ### no working 
    # parser.add_argument('--det_weight', default="./detection/weights/weight_trt/epiton_3_no_nms.trt")  

    ## day time
    # parser.add_argument('--det_weight', default="./detection/weights/weight_trt/epitone_7x_2.trt") ### end2end
    # parser.add_argument('--det_weight', default="./detection/weights/weight_trt/230615_songdo_day_no_nms_2.trt") ### end2end
    # parser.add_argument('--det_weight', default="./detection/weights/weight_trt/230615_songdo_day_no_nms.trt") ### end2end
    ## night time no working
    # parser.add_argument('--det_weight', default="./detection/weights/weight_trt/230615_night_songdo_no_nms.trt") ### end2end  

    parser.add_argument('--seg_weight', default="./segmentation/weights/None_384x640_sim_3.trt") 
    # parser.add_argument('--seg_weight', default="./segmentation/weights/hybridnets_c0_384x640_simplified.trt")  

    parser.add_argument("--end2end", default=True, action="store_true",help="use end2end engine")
    parser.add_argument('--anchor', default='./segmentation/anchors/None_anchors_384x640.npy')
    # parser.add_argument('--nc', type=str, default='10', help='Number of detection classes')
    parser.add_argument('--nc', type=str, default='1', help='Number of detection classes')
    
    day_night_list = ['day','night']
    day_night = day_night_list[0]
    if day_night == 'day':
        print('*'*20)
        print('*** DAY TIME ***')
        print('*'*20)
        # parser.add_argument('--det_weight', default="./detection/weights/weight_trt/epiton_3_with_nms.trt")  
        parser.add_argument('--det_weight', default="./detection/weights/weight_trt/230615_songdo_day_no_nms.trt") ### end2end
    if day_night == 'night':
        print('*'*12)
        print('*** NIGHT TIME ***')
        print('*'*12)
        parser.add_argument('--det_weight', default="./detection/weights/weight_trt/230615_night_songdo_no_nms.trt") ### end2end  
    args = parser.parse_args()
    args = parser.parse_args()

    Camemra_Node = Camemra_Node(args,day_night)
    Camemra_Node.main()


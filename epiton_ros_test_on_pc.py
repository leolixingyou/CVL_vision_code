import os
import time
import math
import numpy as np
import copy
import cv2
import argparse
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image as PLIimage
from io import BytesIO

from detection.det_infer import Predictor
from detection.calibration import Calibration

import rospy
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseArray, Pose

###
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
###

from sort import *

class Camemra_Node:
    def __init__(self,args,day_night):
        rospy.init_node('Camemra_node')
        rate = rospy.Rate(300)
        self.fps_list=[]
        self.args = args
        
        # self.class_names = [ 'person', 'bicycle', 'car', 'motorcycle', 'green3_h', 'bus',
        #                       'red3_h', 'truck', 'yellow3_h', 'green4_h', 'red4_h', 'yellow4_h',
        #                       'redgreen4_h', 'redyellow4_h', 'greenarrow4_h', 'red_v', 'yellow_v', 'green_v','black']
        # self.rgb_day_list = [(148, 108, 138), (207, 134, 240), (138, 88, 55), (26, 99, 86), (38, 125, 80), (72, 80, 57), # 6 
        #                 (118, 23, 200), (117, 189, 183), (0, 128, 128), (60, 185, 90), (20, 20, 150), (13, 131, 204), 
        #                 (30, 200, 200), (43, 38, 105), (104, 235, 178), (135, 68, 28), (140, 202, 15), (67, 115, 220),(30, 80, 30)]
        self.class_names = ['person', 'bicycle','car','bus','motorcycle','truck', 'green', 'red', 'yellow',
                            'red_arrow', 'red_yellow', 'green_arrow','green_yellow','green_right',
                            'warn','black','tl_v', 'tl_p', 'traffic_sign', 'warning', 
                            'tl_bus']
        
        self.rgb_day_list = [(148, 108, 138), (207, 134, 240), (138, 88, 55), (26, 99, 86), (38, 125, 80), (72, 80, 57), # 6 
                        (118, 23, 200), (117, 189, 183), (0, 128, 128), (60, 185, 90), (20, 20, 150), (13, 131, 204), 
                        (30, 200, 200), (43, 38, 105), (104, 235, 178), (135, 68, 28), (140, 202, 15), (67, 115, 220),(30, 80, 30),(30, 80, 30),(30, 80, 30),(30, 80, 30),(30, 80, 30),(30, 80, 30)]


        self.last_observed_light = []
        # self.go_signals = [4, 9, 12, 14, 17]
        self.go_signals = [6, 9, 11, 12, 13]

        self.allowed_unrecognized_frames = 0
        self.last_observed_time = 0
        self.baseline_boxes = [960,270]

        sort_max_age = 15
        sort_min_hits = 2
        sort_iou_thresh = 0.1
        self.sort_tracker_f60 = Sort(max_age=sort_max_age,
                        min_hits=sort_min_hits,
                        iou_threshold=sort_iou_thresh)
        
        local = os.getcwd()
        # print('now is here', local)
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
        
        ### Det Model initiolization
        self.day_night = day_night
        self.det_pred = Predictor(engine_path=args.det_weight , day_night=day_night)
        
        self.pub_od_f60 = rospy.Publisher('/mobinha/perception/camera/bounding_box', PoseArray, queue_size=1)
        rospy.Subscriber('/gmsl_camera/dev/video0/compressed', CompressedImage, self.IMG_f60_callback)

        ##########################
        self.pub_f60_det = rospy.Publisher('/det_result/f60', Image, queue_size=1)
      
        self.bridge = CvBridge()
        self.is_save =False
        self.sup = []
        ##########################
       
        ##########################
        ########filter part#######

        self.real_cls_hist = 0

        ########filter part#######
        ##########################

    def draw_img_filter(self,img, boxes):
        # print(f'boxes is {boxes}')
        height, weight, _ = img.shape
        tl = 3 or round(0.002 * (height + weight) / 2) + 1  # line/font thickness
        tf = max(tl - 1, 1)  # font thickness
        cur_img = copy.copy(img)
        if len(boxes) > 0 :
            box = boxes[0][3]
            cls_id = boxes[0][0]
            score = boxes[0][2]

            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])
            x0 = 0 if  x0 < 0 else x0
            y0 = 0 if  y0 < 0 else y0

            _COLORS = self.rgb_day_list

            c1, c2 = (x0,y0), (x1,y1)
            cv2.rectangle(cur_img, c1, c2, _COLORS[cls_id], thickness=tl, lineType=cv2.LINE_AA)
            tf = max(tl - 1, 1)  # font thickness
            text = '{}:{:.1f}%'.format(self.class_names[cls_id], score * 100)
            t_size = cv2.getTextSize(text, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(cur_img, c1, c2, _COLORS[cls_id], -1, cv2.LINE_AA)  # filled
            cv2.putText(cur_img, text, (c1[0], c1[1] - 2), 0, tl / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)
        img = cur_img
        return img

    def draw_img(self,img,boxes):
        height, weight, _ = img.shape
        tl = 3 or round(0.002 * (height + weight) / 2) + 1  # line/font thickness
        tf = max(tl - 1, 1)  # font thickness
        
        cur_img = copy.copy(img)

        if len(boxes) > 0 :
            for i in range(len(boxes)):

                box = boxes[i][3]
                cls_id = boxes[i][0]
                score = boxes[i][2]

                x0 = int(box[0])
                y0 = int(box[1])
                x1 = int(box[2])
                y1 = int(box[3])
                x0 = 0 if  x0 < 0 else x0
                y0 = 0 if  y0 < 0 else y0

                _COLORS = self.rgb_day_list
                
                c1, c2 = (x0,y0), (x1,y1)
                cv2.rectangle(cur_img, c1, c2, _COLORS[cls_id], thickness=tl, lineType=cv2.LINE_AA)
                tf = max(tl - 1, 1)  # font thickness
                text = '{}:{:.1f}%'.format(self.class_names[cls_id], score * 100)
                t_size = cv2.getTextSize(text, 0, fontScale=tl / 3, thickness=tf)[0]
                c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                cv2.rectangle(cur_img, c1, c2, _COLORS[cls_id], -1, cv2.LINE_AA)  # filled
                cv2.putText(cur_img, text, (c1[0], c1[1] - 2), 0, tl / 3, [255, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

        img = cur_img
        return img

        ##########################
        ########filter part#######

    def filtered_obs(self,traffic_light_obs):
        new_obs = []
        for obs in traffic_light_obs:
            if obs[0] == 18 and self.real_cls_hist != 0:
                obs[0] = self.real_cls_hist
            new_obs.append(obs)
        return new_obs

        ########filter part#######
        ##########################

    def get_one_boxes(self,traffic_light_obs):
        ##########################
        ########filter part#######

        traffic_light_obs = self.filtered_obs(traffic_light_obs)

        ########filter part#######
        ##########################
        if len(traffic_light_obs) >0:
            # print(f'traffic_light_obs is {traffic_light_obs}')            
            boxes = np.array(traffic_light_obs)[:,3]
            distances = [math.sqrt(((box[0] + box[2]) / 2 - self.baseline_boxes[0]) ** 2 + ((box[1] + box[3]) / 2 - self.baseline_boxes[1]) ** 2) for box in boxes]
            areas = [((box[2] - box[0]) * (box[3] - box[1])) for box in boxes]
            weights = [0.6*(distances[x] / max(distances)) + 0.4*(1 - (areas[x] / max(areas)))  for x in range(len(boxes))]
            result_box = [traffic_light_obs[weights.index(min(weights))]]
            return result_box
        else:
            result_box = traffic_light_obs
            return result_box

    def get_traffic_light_objects(self,bbox_f60):
        traffic_light_obs = []

        if len(bbox_f60) > 0:
            for traffic_light in bbox_f60:
                if traffic_light[2] > 0.2:  # if probability exceed 20%
                    traffic_light_obs.append(traffic_light)
        # sorting by size
        traffic_light_obs = self.get_one_boxes(traffic_light_obs)
        return traffic_light_obs

    def IMG_f60_callback(self,msg):
        t1 = time.time()
        if not self.get_f60_new_image:
            np_arr = np.fromstring(msg.data, np.uint8)
            arr_time = time.time()
            #print(f'array_time is ',round((arr_time-t1)*1000,2),' ms')


            # front_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)[:600,:]
            front_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            t2 = time.time()
            #print(f'after decoding is ',round((t2-arr_time)*1000,2),' ms')

            # self.cur_f60_img['img'] = self.calib.undistort(front_img,'f60')
            self.cur_f60_img['img'] = front_img


            self.cur_f60_img['header'] = msg.header
            # fps = 1/(time.time() - t1)
            # self.fps_list.append(fps)
            t3 = time.time()
            # print(f'after undistort is ',round((t3-t2)*1000,2),' ms')

            # average_fps = sum(self.fps_list)/len(self.fps_list)
            # print(f'average FPS is  {average_fps:.2f}')
            
            #print(f'FPS is  {(1/(time.time()-t1))}')
            self.get_f60_new_image = True



    def pose_set(self,bboxes,flag):
        bbox_pose = PoseArray()

        for bbox in bboxes:
            pose = Pose()
            pose.position.x = bbox[0]# box class
            pose.position.y = bbox[1]# box area
            pose.position.z = bbox[2]# box score
            pose.orientation.x = bbox[3][0]# box mid x
            pose.orientation.y = bbox[3][1]# box mid y
            pose.orientation.z = bbox[3][2]# box mid y
            pose.orientation.w = bbox[3][3]# box mid y
            bbox_pose.poses.append(pose)

        if flag == 'f60':
            self.pub_od_f60.publish(bbox_pose)
       
    def det_pubulissher(self,det_img,det_box,flag):
        if flag =='f60':
            det_f60_msg = self.bridge.cv2_to_imgmsg(det_img, "bgr8")#color
            self.pose_set(det_box,flag)
            self.pub_f60_det.publish(det_f60_msg)
    
    def update_tracking(self,box_result,flag):
        update_list = []
        if len(box_result)>0:
            cls_id = np.array(box_result)[:,0]
            areas = np.array(box_result)[:,1]
            scores = np.array(box_result)[:,2]
            boxes = np.array(box_result)[:,3]
            dets_to_sort = np.empty((0,6))

            for i,box in enumerate(boxes):
                x0, y0, x1, y1 = box
                cls_name = cls_id[i]
                dets_to_sort = np.vstack((dets_to_sort, 
                            np.array([x0, y0, x1, y1, scores[i], cls_name])))
            if flag == 'f60':
                tracked_dets = self.sort_tracker_f60.update(dets_to_sort)
                tracks = self.sort_tracker_f60.getTrackers()

            bbox_xyxy = tracked_dets[:,:4]
            categories = tracked_dets[:, 4]

            new_areas = (bbox_xyxy[:,2] - bbox_xyxy[:,0]) * (bbox_xyxy[:,3] - bbox_xyxy[:,1])
            update_list = [[int(categories[x]),new_areas[x],scores[x],bbox_xyxy[x]] for x in range(len(tracked_dets)) ]

        else:
            tracked_dets = self.sort_tracker_f60.update()

        return update_list

    def image_process(self,img,flag):
        if flag == 'f60' :
            # print('f60')
            ### using with vs
            box_result_f60 = self.det_pred.steam_inference(img,conf=0.1, end2end='end2end' ,day_night=self.day_night)
            # print(f'length is {len(box_result_f60)}')

            ### using shell file named 'vision.sh'
            # box_result_f60 = self.det_pred.steam_inference(img,conf=0.1, end2end=args.end2end,day_night=self.day_night)
            # print(f'************box_result_f60 is {box_result_f60}')

            box_result_f60 = self.update_tracking(box_result_f60,flag)
            det_img_60 = self.draw_img(img,box_result_f60)
            # print(f'box_result_f60 is {box_result_f60}')
            tl_boxes = self.get_traffic_light_objects(box_result_f60)
            # print(f'tl_boxes is {tl_boxes}')

        ##########################
        ########filter part#######            
            if len(tl_boxes)>0 :
                self.real_cls_hist = tl_boxes[0][0]
        ########filter part#######
        ##########################

            filter_img_f60 = self.draw_img_filter(det_img_60, tl_boxes)

            self.det_pubulissher(filter_img_f60, tl_boxes,flag)
            

    def main(self):
        rate = rospy.Rate(150)
        while not rospy.is_shutdown():
            tq=time.time()
            if self.get_f60_new_image:
                self.sub_f60_img['img'] = self.cur_f60_img['img']
                orig_im_f60 = copy.copy(self.sub_f60_img['img']) 
                self.image_process(orig_im_f60,'f60')
                self.get_f60_new_image = False
                fps = 1/(time.time() - tq)

                self.fps_list.append(fps)
                average_fps = sum(self.fps_list)/len(self.fps_list)
                # print(f'average FPS is  {average_fps:.2f}')
            rate.sleep()
          

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--end2end", default=False, action="store_true",help="use end2end engine")
    
    day_night_list = ['day','night']
    day_night = day_night_list[0]
    if day_night == 'day':
        # print('*'*12)
        # print('*** DAY TIME ***')
        # print('*'*12)
        # parser.add_argument('--det_weight', defaulssst="./detection/weights/epiton_3_nms.trt")  ### end2end 
        # parser.add_argument('--det_weight', default="./detection/weights/230615_songdo_day_no_nms.trt")  ### end2end 
        # parser.add_argument('--det_weight', default="./detection/weights/yolov7x_flicker_with_nms.trt")  ### end2end  
        # parser.add_argument('--det_weight', default="./detection/weights/weight_trt/yolov7x_flicker.trt")  ### end2end
        # parser.add_argument('--det_weight', default="./detection/weights/weight_trt/incheon_dataset.trt")  ### end2end 
        parser.add_argument('--det_weight', default="./detection/weights/weight_trt/new.trt")  ### end2end  



    if day_night == 'night':
        print('*'*12)
        print('*** NIGHT TIME ***')
        print('*'*12)
        # parser.add_argument('--det_weight', default="./detection/weights/230615_night_songdo_no_nms_2.trt")  ### end2end
        parser.add_argument('--det_weight', default="./detection/weights/230615_night_songdo_no_nms_2.trt")  ### end2end

    args = parser.parse_args()
    
    camemra_node = Camemra_Node(args,day_night)
    camemra_node.main()


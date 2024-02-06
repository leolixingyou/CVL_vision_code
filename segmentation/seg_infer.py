import cv2
import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
from .utils import *
import argparse
# from moviepy.editor import ImageSequenceClip

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class BaseEngine(object):
    def __init__(self, engine_path, input_size, anchor_path, nc):
        self.input_width = input_size[1]
        self.input_height = input_size[0]
        self.model_name = 'hybridnets'
        self.initialize_model(engine_path, anchor_path)
        self.all_ms = []
        self.nc = nc
    
    def initialize_model(self, engine_path, anchor_path='../onnx_models(journal)/regnetY/regnety_008_anchors_256x512.npy', conf_thres=0.3, iou_thres=0.5):
        cuda.init()
        device = cuda.Device(0)
        context = device.make_context()
        cuda.Context.push(context)
        
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(TRT_LOGGER, "")
        # Load a TensorRT engine from file
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        self.engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(engine_data)
        self.stream = cuda.Stream()
        self.context = self.engine.create_execution_context()
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # Read the anchors from the file
        self.anchors = np.squeeze(np.load(anchor_path))

        # Get model input and output shapes
        self.input_shape = self.engine.get_binding_shape(0)
        self.seg_shape = self.engine.get_binding_shape(1)
        self.boxes_shape = self.engine.get_binding_shape(2)
        self.scores_shape = self.engine.get_binding_shape(3)

        print("input_shape: ", self.input_shape)
        print("seg_shape: ", self.seg_shape)
        print("boxes_shape: ", self.boxes_shape)
        print("scores_shape: ", self.scores_shape)

        # Set up the input and output bindings
        self.inputs, self.outputs, self.bindings = [], [], []
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
        print("Bindings: ", self.bindings)

    def infer(self, image):
        self.inputs[0]['host'] = np.ravel(image)
        for input in self.inputs:
            cuda.memcpy_htod_async(input['device'], input['host'], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        for output in self.outputs:
            cuda.memcpy_dtoh_async(output['host'], output['device'], self.stream)
        self.stream.synchronize()
        output  = [output['host'] for output in self.outputs]
        if self.input_width==512 and self.input_height==256:
            seg = np.reshape(output[0], (-1, 6, 256, 512))
            boxes = np.reshape(output[1], (-1, 24552, 4))
            scores = np.reshape(output[2], (-1, 24552, self.nc))
        elif self.input_width==640 and self.input_height==384:
            seg = np.reshape(output[0], (-1, 6, 384, 640))
            # boxes = np.reshape(output[1], (-1, 46035, 4))
            # scores = np.reshape(output[2], (-1, 46035, self.nc))
            
        #print(boxes.shape)
        #print(scores.shape)
        #print(seg.shape)
        
        return {'segmentation': seg}
        # return {'segmentation': seg, 'regression': boxes, 'classification': scores}
        # return [output['host'] for output in self.outputs]

    def inference(self, img_path, conf=0.25):
        origin_img = img_path
        origin_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)
        img = self.prepare_input(origin_img)
        start_time = time.time()
        output = self.infer(img)
        end_time = time.time()
        elapsed_time = end_time - start_time
        self.ms = elapsed_time * 1000
        self.all_ms.append(self.ms)
        #print('Process time:', round(self.ms, 2), 'ms')
        
        self.seg_map  = self.process_output(output, img)

        # self.seg_map, self.filtered_boxes, self.filtered_scores, self.filtered_indexes  = self.process_output(output, img)
        # return self.seg_map, self.filtered_boxes, self.filtered_scores, self.filtered_indexes
      
    def prepare_input(self, image):

        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width,self.input_height))  

        # Scale input pixel values to -1 to 1
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        input_img = ((input_img/ 255.0 - mean) / std)
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis,:,:,:].astype(np.float32)

        return input_tensor
    
    def process_output(self, outputs, img): 

        # Process segmentation map
        seg_map = np.squeeze(np.argmax(outputs["segmentation"], axis=1))

        # Process detections
        # scores = np.squeeze(outputs["classification"])
        # boxes = np.squeeze(outputs["regression"])

        # filtered_boxes, filtered_scores, filtered_indexes =  self.process_detections(scores, boxes)

        return seg_map
        # return seg_map, boxes, scores
        # return seg_map, filtered_boxes, filtered_scores, filtered_indexes
    
    def process_detections(self, scores, boxes):
        # print("Scores: ", scores.shape)
        # print("Boxes: ", boxes.shape)
        max_indexes = np.squeeze(np.argmax(scores, axis=-1))
        scores = np.squeeze(np.max(scores, axis=-1, keepdims=True))
        # print("Scores: ", scores.shape)

        transformed_boxes = transform_boxes(boxes, self.anchors)

        # Filter out low score detections
        filtered_boxes = transformed_boxes[scores>self.conf_thres]
        filtered_scores = scores[scores>self.conf_thres]
        filtered_indexes = max_indexes[scores>self.conf_thres]

        # Resize the boxes with image size
        filtered_boxes[:,[0,2]] *= self.img_width/self.input_width
        filtered_boxes[:,[1,3]] *= self.img_height/self.input_height

        # Perform nms filtering
        filtered_boxes, filtered_scores, filtered_indexes = nms_fast(filtered_boxes, filtered_scores, filtered_indexes, self.iou_thres)

        # print("Filtered boxes: ", filtered_boxes.shape)
        # print("Filtered scores: ", filtered_scores.shape)
        # print("Filtered indexes: ", filtered_indexes.shape)

        return filtered_boxes, filtered_scores, filtered_indexes

    def draw_boxes(self, image, text=True):

        return util_draw_detections(self.filtered_boxes, self.filtered_scores, self.filtered_indexes, image, text, self.nc)

    def draw_segmentation(self, image, alpha = 0.5):
        print(self.seg_map.shape)
        return util_draw_seg(self.seg_map, image, alpha)

    def draw_2D(self, image, alpha = 0.5, text=True):
        # self.context.pop()
        front_view = self.draw_segmentation(image, alpha)
        # front_view, mask = self.draw_segmentation(image, alpha)
        # return self.draw_segmentation(image, alpha)
        return front_view, self.all_ms
        # return self.draw_boxes(front_view, text), self.all_ms


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MultiHeadNet: Edge Device Oriented Multi Task Network - Shokhrukh, Shakhboz')
    parser.add_argument('-i', '--input', type=str, help='The input file path')
    parser.add_argument('-o', '--output', type=str, help='Output name')
    # parser.add_argument('-e', '--engine', type=str, required=True, help='The TensorRT engine file path')
    parser.add_argument('-e', default="./segmentation/weights/None_384x640_sim_3.trt")  
    parser.add_argument('-a', '--anchor', type=str, required=True, help='The anchors numpy file path')
    parser.add_argument('--dtype', type=str, default="FP16", help='The anchors numpy file path')
    parser.add_argument('--input_mode', type=str, default="large", help='Model input mode')
    parser.add_argument('--nc', type=str, default='1', help='Number of detection classes')
    
    args = parser.parse_args()

    print('Start')
    im_seq = []

    

    model = args.engine if args.engine else '../onnx_models(journal)/regnetY/regnety_008_256x512_simplified_int8.trt'
    input_path = args.input if args.input else 'image.jpg'
    anchors = args.anchor if args.anchor else None
    input_size = (384, 640)
    nc = int(args.nc)
    # if nc != 10 and nc != 13:
    #     print(nc)
    #     print('INCORRECT NUMBER OF CLASSES ENTERED')
    #     import sys
    #     sys.exit()
    if args.input_mode=='small':
        input_size = (256, 512) 
    
    try:
        pred = BaseEngine(model, input_size, anchors, nc)
    except:
        cuda.Context.pop()


    try:
        if input_path.endswith('.mp4'):
            output_path = args.output if args.output else 'output.avi'
            # #inference on video input
    #        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #        writer = cv2.VideoWriter(output_path, fourcc, 30.0, (640, 384))
            cap = cv2.VideoCapture(input_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                seg_map, filtered_boxes, filtered_scores, filtered_indexes = pred.inference(frame)
                combined, mask, t_ms = pred.draw_2D(frame)
                # cv2.putText(combined, 'Precision: {}'.format(args.dtype), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('frame', combined)
                cv2.waitKey(1)
                combined = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
                im_seq.append(combined)
    #            writer.write(combined)
    #            if cv2.waitKey(1) & 0xFF == ord('q'):
    #                break
    #        writer.release()
            cv2.destroyAllWindows()
            cap.release()


        
        elif input_path.endswith('.jpg') or input_path.endswith('.png'):
            output_path = args.output if args.output else 'output.jpg'
            #inference on image input
            image = cv2.imread(input_path)
            seg_map = pred.inference(image)
            combined, mask, t_ms = pred.draw_2D(image)
            cv2.imwrite(output_path, combined)
            cv2.imwrite(f'{output_path[:-4]}_mask.png', mask)
        # print(f"Average processing time: {sum(t_ms[9:])/len(t_ms[9:])} ms / {1000/(sum(t_ms)/len(t_ms))} fps")
    except:
        pred.context.pop()

        # cuda.Context.pop()
        
    
    if input_path.endswith('.mp4'):
        clip = ImageSequenceClip(im_seq, fps=24)
        clip.write_videofile(output_path.replace('.avi', '.mp4'))
    # else:
    #     import matplotlib.pyplot as plt
    #     #show mas in plot
    #     plt.imshow(mask)
    #     plt.show()
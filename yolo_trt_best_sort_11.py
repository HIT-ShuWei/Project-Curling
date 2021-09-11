"""
An example that uses TensorRT's Python api to make inferences.
"""
import ctypes
import os
import shutil
import random
import sys
import threading
import time
import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import torch
import torchvision
import argparse
import time
#sort
from sort import *
import numpy
 
CONF_THRESH = 0.5  #0.5
IOU_THRESHOLD = 0.4  #0.4
mot_tracker_15 = Sort()
mot_tracker_16 = Sort()
 
def get_img_path_batches(batch_size, img_dir):
    ret = []
    batch = []
    for root, dirs, files in os.walk(img_dir):
        for name in files:
            if len(batch) == batch_size:
                ret.append(batch)
                batch = []
            batch.append(os.path.join(root, name))
    if len(batch) > 0:
        ret.append(batch)
    return ret
 
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    param: 
        x:      a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return
    """
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )
 
 
class YoLov5TRT(object):
    """
    description: A YOLOv5 class that warps TensorRT ops, preprocess and postprocess ops.
    """
    
    def __init__(self, engine_file_path):
        # Create a Context on this device,
        self.ctx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)
        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()
 
        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []
 
        for binding in engine:
            print('bingding:', binding, engine.get_binding_shape(binding))
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)
        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.batch_size = engine.max_batch_size
 
    def infer(self, input_image_path):
        threading.Thread.__init__(self)
        # Make self the active context, pushing it on top of the context stack.
        self.ctx.push()
        self.input_image_path = input_image_path
        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        # Do image preprocess
        batch_image_raw = []
        batch_origin_h = []
        batch_origin_w = []
        batch_input_image = np.empty(shape=[self.batch_size, 3, self.input_h, self.input_w])
 
        input_image, image_raw, origin_h, origin_w = self.preprocess_image(input_image_path
                                                                           )
 
 
        batch_origin_h.append(origin_h)
        batch_origin_w.append(origin_w)
        np.copyto(batch_input_image, input_image)
        batch_input_image = np.ascontiguousarray(batch_input_image)
 
        # Copy input image to host buffer
        np.copyto(host_inputs[0], batch_input_image.ravel())
        start = time.time()
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # Run inference.
        context.execute_async(batch_size=self.batch_size, bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        # Synchronize the stream
        stream.synchronize()
        end = time.time()
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
        # Here we use the first row of output in that batch_size = 1
        output = host_outputs[0]
        # Do postprocess
        result_boxes, result_scores, result_classid = self.post_process(
            output, origin_h, origin_w)
        
        ##### ------sort------ #####by zl
        result_boxes=numpy.array(result_boxes)
        pred_boxes_15=[]
        pred_boxes_16=[]
        for q in range(len(result_boxes)):
            print(q,int(result_classid[q]))
            if int(result_classid[q])==15:
                pred_boxes_15.append(result_boxes[q]) 
            if int(result_classid[q])==16:
                pred_boxes_16.append(result_boxes[q])
                #print(pred_boxes_16)   
        pred_boxes_15=numpy.array(pred_boxes_15)
        pred_boxes_16=numpy.array(pred_boxes_16)         
        if len(pred_boxes_15) == 0:
            pred_boxes_15 = np.empty((0,5))
        if len(pred_boxes_16) == 0:
            pred_boxes_16 = np.empty((0,5))
        print(pred_boxes_16) 
        track_bbs_ids_15 = mot_tracker_15.update(pred_boxes_15) 
        print("track_bbs_ids_15:",track_bbs_ids_15) 
        track_bbs_ids_16 = mot_tracker_16.update(pred_boxes_16) 
        print("track_bbs_ids_16:",track_bbs_ids_16)
        for i in range (len(track_bbs_ids_15)):
            h=track_bbs_ids_15[i, :4]
            ID = int(track_bbs_ids_15[i,4])  
            label="{}:{}".format(categories[15],ID) 
            plot_one_box(h,image_raw,color=(0,0,255),label=label) 
        for j in range (len(track_bbs_ids_16)):
            h1=track_bbs_ids_16[j, :4]
            ID1 = int(track_bbs_ids_16[j,4])  
            label="{}:{}".format(categories[16],ID1) 
            plot_one_box(h1,image_raw,color=(0,255,180),label=label) 
        #Draw rectangles and labels on the original image
        #for j in range(len(result_boxes)):
        #    box = result_boxes[j, :4]
        #    plot_one_box(
        #        box,
        #        image_raw,
        #        label="{}:{:.2f}".format(categories[int(result_classid[j])], result_scores[j]),
        #    )

        # track bbox message uesd for UDP
        track_bbs_UDP = (track_bbs_ids_15, track_bbs_ids_16)
        return image_raw, end - start, track_bbs_UDP
 
    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
        
    def get_raw_image(self, image_path_batch):
        """
        description: Read an image from image path
        """
        for img_path in image_path_batch:
            yield cv2.imread(img_path)
        
    def get_raw_image_zeros(self, image_path_batch=None):
        """
        description: Ready data for warmup
        """
        for _ in range(self.batch_size):
            yield np.zeros([self.input_h, self.input_w, 3], dtype=np.uint8)
 
    def preprocess_image(self, input_image_path):
        """
        description: Convert BGR image to RGB,
                     resize and pad it to target size, normalize to [0,1],
                     transform to NCHW format.
        param:
            input_image_path: str, image path
        return:
            image:  the processed image
            image_raw: the original image
            h: original height
            w: original width
        """
        image_raw = input_image_path
        h, w, c = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        # Calculate widht and height and paddings
        r_w = self.input_w / w
        r_h = self.input_h / h
        if r_h > r_w:
            tw = self.input_w
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((self.input_h - th) / 2)
            ty2 = self.input_h - th - ty1
        else:
            tw = int(r_h * w)
            th = self.input_h
            tx1 = int((self.input_w - tw) / 2)
            tx2 = self.input_w - tw - tx1
            ty1 = ty2 = 0
        # Resize the image with long side while maintaining ratio
        image = cv2.resize(image, (tw, th))
        # Pad the short side with (128,128,128)
        image = cv2.copyMakeBorder(
            image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
        )
        image = image.astype(np.float32)
        # Normalize to [0,1]
        image /= 255.0
        # HWC to CHW format:
        image = np.transpose(image, [2, 0, 1])
        # CHW to NCHW format
        image = np.expand_dims(image, axis=0)
        # Convert the image to row-major order, also known as "C order":
        image = np.ascontiguousarray(image)
        return image, image_raw, h, w
 
    def xywh2xyxy(self, origin_h, origin_w, x):
        """
        description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        param:
            origin_h:   height of original image
            origin_w:   width of original image
            x:          A boxes tensor, each row is a box [center_x, center_y, w, h]
        return:
            y:          A boxes tensor, each row is a box [x1, y1, x2, y2]
        """
        y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
        r_w = self.input_w / origin_w
        r_h = self.input_h / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h
 
        return y
 
    def post_process(self, output, origin_h, origin_w):
        """
        description: postprocess the prediction
        param:
            output:     A tensor likes [num_boxes,cx,cy,w,h,conf,cls_id, cx,cy,w,h,conf,cls_id, ...] 
            origin_h:   height of original image
            origin_w:   width of original image
        return:
            result_boxes: finally boxes, a boxes tensor, each row is a box [x1, y1, x2, y2]
            result_scores: finally scores, a tensor, each element is the score correspoing to box
            result_classid: finally classid, a tensor, each element is the classid correspoing to box
        """
        # Get the num of boxes detected
        num = int(output[0])
        # Reshape to a two dimentional ndarray
        pred = np.reshape(output[1:], (-1, 6))[:num, :]
        # to a torch Tensor
        pred = torch.Tensor(pred).cuda()
        # Get the boxes
        boxes = pred[:, :5]
        # Get the scores
        scores = pred[:, 4]
        # Get the classid
        classid = pred[:, 5]
        # Choose those boxes that score > CONF_THRESH
        si = scores > CONF_THRESH
        boxes = boxes[si, :]
        scores = scores[si]
        classid = classid[si]
        # Trandform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
        boxes[:, :4] = self.xywh2xyxy(origin_h, origin_w, boxes[:, :4])
        # Do nms
        indices = torchvision.ops.nms(boxes[:, :4], scores, iou_threshold=IOU_THRESHOLD).cpu()
        result_boxes = boxes[indices, :].cpu()
        result_scores = scores[indices].cpu()
        result_classid = classid[indices].cpu()
        return result_boxes, result_scores, result_classid
 
 
class inferThread(threading.Thread):
    def __init__(self, yolov5_wrapper):
        threading.Thread.__init__(self)
        self.yolov5_wrapper = yolov5_wrapper
    def infer(self , frame):
        batch_image_raw, use_time = self.yolov5_wrapper.infer(frame)

        # for i, img_path in enumerate(self.image_path_batch):
        #     parent, filename = os.path.split(img_path)
        #     save_name = os.path.join('output', filename)
        #     # Save image
        #     cv2.imwrite(save_name, batch_image_raw[i])
        # print('input->{}, time->{:.2f}ms, saving into output/'.format(self.image_path_batch, use_time * 1000))
        return batch_image_raw,use_time
 
class warmUpThread(threading.Thread):
    def __init__(self, yolov5_wrapper):
        threading.Thread.__init__(self)
        self.yolov5_wrapper = yolov5_wrapper
 
    def run(self):
        batch_image_raw, use_time = self.yolov5_wrapper.infer(self.yolov5_wrapper.get_raw_image_zeros())
        print('warm_up->{}, time->{:.2f}ms'.format(batch_image_raw[0].shape, use_time * 1000))
 
 
 
if __name__ == "__main__":
    # load custom plugins  
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', nargs='+', type=str, default="build/yolov5s.engine", help='.engine path(s)')
    parser.add_argument('--save', type=int, default=0, help='save?')
    opt = parser.parse_args()
    PLUGIN_LIBRARY = "build/libmyplugins.so"
    engine_file_path = opt.engine
 
    ctypes.CDLL(PLUGIN_LIBRARY)
 
    # load coco labels
 
    categories = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','red','yellow']
    # a YoLov5TRT instance
    mot_tracker = Sort()   #create class sort
    yolov5_wrapper = YoLov5TRT(engine_file_path)
    cap = cv2.VideoCapture(0)
    #cap=cv2.imread("/home/zl/Downloads/tensorrtx-master/yolov5/images/train2017/")
    try:
        thread1 = inferThread(yolov5_wrapper)
        thread1.start()
        thread1.join()
        while 1:
            _,frame = cap.read()
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #frame = Image.fromarray(np.uint8(frame))
            start=time.time()

            # img, t =thread1.infer(frame)
            # add track_message
            img, t, (track_bbs_red, track_bbs_yellow) =thread1.infer(frame)

            #----------------------------------------------#
            #       biuld a UPD clinet to send message
            #----------------------------------------------#
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

            track_bbs_UDP = np.array(track_bbs_red, track_bbs_yellow)
            track_bbs_UDP.tostring()    # the first one is red, then yellow
            
            s.sendto(track_bbs_UDP, ('127.0.0.3', 9999))    


            
            #fourcc = cv2.VideoWriter_fourcc(*'XVID')
            #cv2.VideoWriter('1.avi', fourcc, 30.0, (640,480))   #save video
            end=time.time()
            a=float(1/(end-start))
            print(a)
            cv2.imshow("result", img)
            if cv2.waitKey(1) & 0XFF == ord('q'):  # 1 millisecond
                break
        ctx.pop()
 
    finally:
        #destroy the instance
        cap.release()
        
        #cudaFree(ctx)
        cv2.destroyAllWindows()
        yolov5_wrapper.destroy()

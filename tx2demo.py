from models import *
from utils import *
import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image
from collections import deque
import time
import warnings

config_path='cfg/yolov3.cfg'
weights_path='backup_0116/yolov3_6000.weights'
class_path='cfg/Hand.names'
img_size=416
conf_thres=0.8
nms_thres=0.4
# Load model and weights
model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
model.cuda()
model.eval()
classes = utils.load_classes(class_path)
Tensor = torch.cuda.FloatTensor

warnings.simplefilter('ignore')

def detect_image(img):
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms=transforms.Compose([transforms.Resize((imh,imw)),
         transforms.Pad((max(int((imh-imw)/2),0), 
              max(int((imw-imh)/2),0), max(int((imh-imw)/2),0),
              max(int((imw-imh)/2),0)), (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)
    return detections[0]


cmap = plt.get_cmap('tab20b')
colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

# initialize Sort object and video capture
import cv2
from sort import *

vid = cv2.VideoCapture(1)
mot_tracker = Sort()
li_max = 10
li_del = 0
tests = []
flscr = 'fullscreen'
circle_list = {}
#color_list = {}
#time_counter = {}
check_id = deque()
centers = None
#cv2.namedWindow(flscr, cv2.WND_PROP_FULLSCREEN)
#cv2.setWindowProperty(flscr, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while(vid.isOpened()):
    ret, frame = vid.read()
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(frame)
    detections = detect_image(pilimg)
    
    #frame = np.zeros((480, 640, 3))
    img = np.array(pilimg)
    mot_tracker.img_shape = [img.shape[0], img.shape[1]]
    mot_tracker.pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    mot_tracker.pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    mot_tracker.unpad_h = img_size - mot_tracker.pad_y
    mot_tracker.unpad_w = img_size - mot_tracker.pad_x

    if detections is not None:
        tracked_objects = mot_tracker.update(detections.cpu())
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)

        for i, trk in enumerate(mot_tracker.trackers):
            color = colors[int(trk.id+1) % len(colors)]
            color = [i * 255 for i in color]
            trk.color = color

            cls = classes[int(trk.objclass)]

            d = trk.get_state()[0]
            box_w = int(((d[2] - d[0]) / mot_tracker.unpad_w) * img.shape[1])
            box_h = int(((d[3] - d[1]) / mot_tracker.unpad_h) * img.shape[0])
            x1 = int(((d[0] - mot_tracker.pad_x / 2) / mot_tracker.unpad_w) * img.shape[1])
            y1 = int(((d[1] - mot_tracker.pad_y / 2) / mot_tracker.unpad_h) * img.shape[0])
            
            cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), (255,255,255), 4)
            #cv2.rectangle(frame, (x1, y1-35), (x1+len(cls)*19+60, y1), trk.color, -1)
            cv2.putText(frame, cls + "-" + str(int(trk.id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,2555,255), 3)

            if not trk.id in circle_list:
                circle_list[trk.id] = []
            #if not trk.id in color_list:
            #    color_list[trk.id] =[]
            #if not trk.id in time_counter:
            #    time_counter[trk.id] = [] 
            if trk.cir_x is not None and trk.cir_y is not None:
                if cls == 'Write':
                    circle_list[trk.id].append((trk.cir_x, trk.cir_y))
            #        color_list[trk.id].append(trk.color)
            #        time_counter[trk.id].append(time.perf_counter())

    for box_id in circle_list.keys():
        for i in range(len(circle_list[box_id])):
            cv2.circle(frame, circle_list[box_id][i], 3, (255,255,255), -1)
            if i > 0:
                cv2.line(frame, circle_list[box_id][i-1], circle_list[box_id][i], (255,255,255), 6)
                #if int(time_counter[box_id][i]) > 5:
                #    del circle_list[box_id][i]
                #    del time_counter[box_id]
                #    print('deleted'

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow(flscr, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    if cv2.waitKey(1) & 0xFF == ord('c'):
        circle_list.clear()
        print('deleted')
 

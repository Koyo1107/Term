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
circle_list = deque()
color_list = deque()
B = []
G = []
R = []
centers = None
#cv2.namedWindow(flscr, cv2.WND_PROP_FULLSCREEN)
#cv2.setWindowProperty(flscr, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while(vid.isOpened()):
#for ii in range(40):
    ret, frame = vid.read()
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(frame)
    detections = detect_image(pilimg)
    
    img = np.array(pilimg)
    mot_tracker.img_shape = [img.shape[0], img.shape[1]]
    mot_tracker.pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    mot_tracker.pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    mot_tracker.unpad_h = img_size - mot_tracker.pad_y
    mot_tracker.unpad_w = img_size - mot_tracker.pad_x
    #print('tx2demo.py before if run')

    if detections is not None:
        tracked_objects = mot_tracker.update(detections.cpu())
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        #print('mot_tracker run')            

        for trk in mot_tracker.trackers:
            color = colors[int(trk.id+1) % len(colors)]
            color = [i * 255 for i in color]
            trk.color = color

            cls = classes[int(trk.objclass)]

            #print('cls check run')
            d = trk.get_state()[0]
            box_w = int(((d[2] - d[0]) / mot_tracker.unpad_w) * img.shape[1])
            box_h = int(((d[3] - d[1]) / mot_tracker.unpad_h) * img.shape[0])
            x1 = int(((d[0] - mot_tracker.pad_x / 2) / mot_tracker.unpad_w) * img.shape[1])
            y1 = int(((d[1] - mot_tracker.pad_y / 2) / mot_tracker.unpad_h) * img.shape[0])
            
            #print('trk_color', trk.color) 
            cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h),trk.color, 4)
            #cv2.rectangle(frame, (x1, y1-35), (x1+len(cls)*19+60, y1), trk.color, -1)
            #cv2.putText(frame, cls + "-" + str(int(trk.id)),(x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)

            if cls == 'Write':
                #circle_list.appendleft(trk.centers)
                circle_list.append((trk.cir_x, trk.cir_y))
                color_list.append(trk.color)
                #B.append(trk.col_b)
                #G.append(trk.col_g)
                #R.append(trk.col_r)
                #print('color_list', B, G, R)

    for i in range(len(circle_list)):
        #if color_list[i] is not None:
        #print('color_data', (B[i], G[i], R[i]))
        cv2.circle(frame, circle_list[i], 10, color_list[i], -1)
    
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow(flscr,frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import os
import shutil 

from models.inception_resnet_v1 import InceptionResnetV1
from models.yolo_v3 import Darknet
from utils.datasets import *
from utils.utils import *
import streamlit as st

data_dir = './data/test_images'
out = './output'
workers = 0 # 0 for Windows
margin = 5
conf_thres = 0.3
iou_thres = 0.4
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# ### Preprocess photos to get face pictures
def collate_pil(x): 
    out_x, out_y = [], [] 
    for xx, yy in x: 
        out_x.append(xx) 
        out_y.append(yy) 
    return out_x, out_y 
def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor) / 255.0
    return processed_tensor
photos = datasets.ImageFolder(data_dir)
photos.samples = [
    (p, p.replace(data_dir, data_dir + '_cropped'))
        for p, _ in photos.samples
]
loader = DataLoader(
    photos,
    num_workers=workers,
    batch_size=1,
    collate_fn=collate_pil
)

# Initialize yolo model
yolo = Darknet('models/yolov3-tiny-1cls.cfg', img_size=416)
yolo.load_state_dict(torch.load('models/yolov3-tiny-1cls-20191229.pt', map_location=device)['model'])
yolo.to(device).eval()
print('Cropping images...')
for i, (img0, path) in enumerate(loader):
    img0 = cv2.cvtColor(np.asarray(img0[0]),cv2.COLOR_RGB2BGR)  
    img = letterbox(img0, new_shape=416)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to fp16/fp32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    img = torch.from_numpy(img).to(device)
    img = img.unsqueeze(0)
    pred = yolo(img)[0]
    pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres)
    det = pred[0]
    if det is not None and len(det):
        det = scale_coords(img.shape[2:], det[4:], img0.shape)
        for box in det[:, :4]:
            extract_face(img0, box[0], margin=margin, save_path=path[0])
    # print('Image ',i,'' done! ')
print('Done cropping pictures of pre-set photos')
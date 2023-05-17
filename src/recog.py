import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import os
import shutil 

from models.inception_resnet_v1 import InceptionResnetV1
from facenet_pytorch import MTCNN
from models.yolo_v3 import Darknet
from utils.datasets import *
from utils.utils import *
import streamlit as st

# # APP
st.title("Face Recognition\nby Ruihan Zhang")

### General Arguments
data_dir = './data/test_images'
out = './output'
workers = 0 # 0 for Windows
margin = 5
conf_thres = 0.3
iou_thres = 0.4
source = './' # '0' for local camera; Or a path to a folder of images
if st.button('Camera Mode'):
    source = '0'
else:
    source = st.text_input('Or you can input an image folder:', value='', key=None)
# Initialize CUDA device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# Initialize yolo model
yolo = Darknet('models/yolov3-tiny-1cls.cfg', img_size=416)
yolo.load_state_dict(torch.load('models/yolov3-tiny-1cls-20191229.pt', map_location=device)['model'])
yolo.to(device).eval()
# Initialize Resnet
resnet = InceptionResnetV1(
    classify=False,
    pretrained='vggface2',
).to(device)
resnet.eval()

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

## Preprocess face pictures into face features
def image_standardization(image_tensor):
    processed_tensor = (image_tensor) / 255.0
    return processed_tensor

trans = transforms.Compose([
    np.float32,
    fixed_image_standardization
])
photos = datasets.ImageFolder(data_dir + '_cropped', transform=trans)
loader = DataLoader(
    photos,
    num_workers=workers,
    batch_size=1,
    collate_fn=collate_pil
)
# Get Embeddings
embeddings = []
names = []
for i, (img, nameid) in enumerate(loader):
    t = time.time()
    img = np.stack(img, 0)
    img=np.swapaxes(img, 2, 3)
    img=np.swapaxes(img, 1, 2)
    img = torch.from_numpy(img).to(device)
    emb = resnet(img).detach().cpu().numpy()
    embeddings.append(emb[0])
    names.append(photos.classes[nameid[0]])
print('Done getting embeddings for faces! Start running on source')
uniqNames = list(set(names))
st.write(uniqNames)


### RUN!
save_img=False
view_img=False
save_txt=False
webcam = source == '0'
if webcam:
    view_img = True
    torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams(source, img_size=416)
else:
    save_img = True
    dataset = LoadImages(source, img_size=416)

t0 = time.time()
for path, img, im0s, in dataset:
    t = time.time()
    # Get detections
    img = torch.from_numpy(img).to(device)
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    pred = yolo(img)[0]
    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres)
    # Process detections
    for i, det in enumerate(pred):  # detections of one image
        if webcam:  # batch_size >= 1
            p, s, im0 = path[i], '%g: ' % i, im0s[i]
        else:
            p, s, im0 = path, '', im0s
        save_path = str(Path(out) / Path(p).name)
        s = 'Image shape(%gx%g) : ' % img.shape[2:] # print string: image information
        if det is not None and len(det):
            # try:
            # Rescale boxes from img_size to im0 size # xyxy
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape)
            #  Crop image into aligned faces
            faces = []
            for box in det[:, :4]:
                face = extract_face(im0, box, margin=margin)#, save_path=out+'/temp.png') # get tensors of faces
                # Second-Stage Ressnet classification
                face = torch.from_numpy(np.array(face)).to(device)
                face = face.unsqueeze(0)
                emb = resnet(face).detach().cpu().numpy()
                idx = np.argmin(np.linalg.norm(embeddings - emb, axis=1,ord = 1))
                label = '%s' % names[idx]
                s += label + '(%d)' % idx  # add to string
                plot_one_box(box, im0, label=label, color=[50,50,50])
                # Print time (inference + NMS)
            fps = 1.0 / (time.time() - t)
            print('%s detected!  (FPS: %f )' % (s, fps))
            # except:
            #     print('FaceSizeError')

        # Stream results
        if view_img:
            cv2.imshow(p, im0)
            if cv2.waitKey(1) == ord('q'):  # q to quit
                raise StopIteration

        # Save results (image with detections)
        if save_img:
            if dataset.mode == 'images':
                cv2.imwrite(save_path, im0)
                st.image(im0, channels = 'BGR', caption = s)

if save_txt or save_img:
    print('Results saved to %s' % os.getcwd() + os.sep + out)
print('Done. (%.3fs)' % (time.time() - t0))
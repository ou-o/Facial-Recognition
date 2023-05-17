from facenet_pytorch import training
from models.inception_resnet_v1 import InceptionResnetV1
from models.yolo_v3 import Darknet

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import os
import matplotlib
from utils.datasets import *
from utils.utils import *
import shutil  
shutil.rmtree('./data/test_images_cropped')  
### PREPROCESS
def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor) / 255.0
    return processed_tensor

torch.cuda.empty_cache()

data_dir = './data/test_images'

batch_size = 32
epochs = 20
workers = 0
margin = 20

conf_thres = 0.3 
iou_thres = 0.4

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# Initialize yolo model (Finetuning and First-Stage)
yolo = Darknet('models/yolov3-tiny-1cls.cfg', img_size=416)
yolo.load_state_dict(torch.load('models/yolov3-tiny-1cls-20191229.pt', map_location=device)['model'])
yolo.to(device).eval()

resnet = InceptionResnetV1(
    classify=True,
    pretrained='vggface2',
    num_classes= len(photos.class_to_idx)
).to(device)

### Preprocessing
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
print('Done Preprocessing! Start Running on Source..')





for k,v in resnet.named_parameters(): # Freeze Graph
    if k=='logits.weight' or k=='logits.bias':
        v.requires_grad = True
    else:
        v.requires_grad = False
    #print(k, v.requires_grad)
    
# #### Define optimizer, scheduler, dataset, and dataloader
optimizer = optim.Adam(resnet.parameters(), lr=0.001)
scheduler = MultiStepLR(optimizer, [5, 10])
trans = transforms.Compose([
    np.float32,
    transforms.ToTensor(),
    fixed_image_standardization
])
photos = datasets.ImageFolder(data_dir + '_cropped', transform=trans)
# Get names and colors
names = photos.classes
print(names)
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
img_inds = np.arange(len(photos))
np.random.shuffle(img_inds)
train_inds = img_inds#[:int(0.8 * len(img_inds))]
val_inds = img_inds#[int(0.8 * len(img_inds)):]
print('train_inds')
print(train_inds)
print('val_inds')
print(val_inds)

train_loader = DataLoader(
    photos,
    num_workers=workers,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(train_inds)
)
val_loader = DataLoader(
    photos,
    num_workers=workers,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(val_inds)
)


# #### Define loss and evaluation functions
loss_fn = torch.nn.CrossEntropyLoss()
metrics = {
    'fps': training.BatchTimer(),
    'acc': training.accuracy
}


# #### Train model
writer = SummaryWriter()
writer.iteration, writer.interval = 0, 10

print('\n\nInitial')
print('-' * 10)
resnet.eval()
training.pass_epoch(
    resnet, loss_fn, val_loader,
    batch_metrics=metrics, show_running=True, device=device,
    writer=writer
)

for epoch in range(epochs):
    print('\nEpoch {}/{}'.format(epoch + 1, epochs))
    print('-' * 10)

    resnet.train()
    training.pass_epoch(
        resnet, loss_fn, train_loader, optimizer, scheduler,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )

    resnet.eval()
    training.pass_epoch(
        resnet, loss_fn, val_loader,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )

writer.close()



### START

# Set Dataloader
source = '0'
save_img=False
view_img=False
save_txt=False
webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
if webcam:
    view_img = True
    torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams(source, img_size=416)
else:
    save_img = True
    dataset = LoadImages(source, img_size=416)

# Run Detections and Recognitions
t0 = time.time()
for path, img, im0s, in dataset:
    t = time.time()
    torch.cuda.empty_cache()

    # Get detections
    img = torch.from_numpy(img).to(device)
    # if img.ndimension() == 3:
    #     img = img.unsqueeze(0)
    # First-Stage YOLO detections
    pred = yolo(img)[0]
    # Apply NMS
    pred = non_max_suppression(pred, 0.3, 0.4)

    # Process detections

    for i, det in enumerate(pred):  # detections of one image
        if webcam:  # batch_size >= 1
            p, s, im0 = path[i], '%g: ' % i, im0s[i]
        else:
            p, s, im0 = path, '', im0s
        #save_path = str(Path(out) / Path(p).name)
        s = 'Image shape(%gx%g) : ' % img.shape[2:] # print string: image information
        if det is not None and len(det):
            try:
                # Rescale boxes from img_size to im0 size # xyxy
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape)
                #  Crop image into aligned faces
                faces = []
                for box in det[:, :4]:
                    face = extract_face(im0, box, margin=margin,save_path='./output/Crop_tmp.png') # get tensors of faces
                    # Second-Stage Ressnet classification
                    face = torch.from_numpy(np.array(face)).to(device)
                    # if img.ndimension() == 3:
                    face = face.unsqueeze(0)
                    logit = resnet(face).detach().cpu().numpy()
                    idx = np.argmax(logit)
                    label = '%s' % names[idx]
                    s += label  # add to string
                    plot_one_box(box, im0, label=label, color=[50,50,50])
                    # Print time (inference + NMS)
                print('%s detected! (%.3fs)' % (s, time.time() - t))

                    # # Write results
                    # for *xyxy, conf, cls in det:
                    #     if save_txt:  # Write to file
                    #         with open(save_path + '.txt', 'a') as file:
                    #             file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                    #     if save_img or view_img:  # Add bbox to image
                    #         label = '%s %.2f' % (names[int(c)], conf)
                    #         plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
            except:
                print('Error')

        # Stream results
        if view_img:
            cv2.imshow(p, im0)
            if cv2.waitKey(1) == ord('q'):  # q to quit
                raise StopIteration

        # Save results (image with detections)
        if save_img:
            if dataset.mode == 'images':
                cv2.imwrite(save_path, im0)


if save_txt or save_img:
    print('Results saved to %s' % os.getcwd() + os.sep + out)
print('Done. (%.3fs)' % (time.time() - t0))
import argparse
import cv2
from sys import platform
from models.yolo_v3 import Darknet
from models.inception_resnet_v1 import InceptionResnetV1
from utils.datasets import *
from utils.utils import *

def detect(save_txt=False, save_img=False):
    out, source, weights, view_img = opt.output, opt.source, opt.weights, opt.view_img
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize yolo model (First-Stage)
    yolo = Darknet(opt.cfg, img_size=416)
    yolo.load_state_dict(torch.load(weights, map_location=device)['model'])
    yolo.to(device).eval()

    # Initialize resnet model (Second-stage)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)  # initialize

    # Set Dataloader
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=416)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=416)

    # Get names and colors
    names = load_classes(opt.names)
    face_id = []
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run Detections and Recognitions
    t0 = time.time()
    for path, img, im0s, in dataset:
        t = time.time()
        torch.cuda.empty_cache()

        # Get detections
        img = torch.from_numpy(img).to(device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
            torch.unsqueeze
        # First-Stage YOLO detections
        pred = yolo(img)[0]
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres)

        # Process detections
        for i, det in enumerate(pred):  # detections of one image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i]
            else:
                p, s, im0 = path, '', im0s
            save_path = str(Path(out) / Path(p).name)
            s = 'Image shape(%gx%g) : ' % img.shape[2:] # print string: image information
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size # xyxy
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape)
                faces = []
                #  Crop image into aligned faces
                for j, box in enumerate(det[:, :4]):
                    face = extract_face(im0, box, margin=0) # get tensors of cropped image
                #     faces.append(face)
                # faces = torch.from_numpy(np.array(faces)).to(device)
                # # Second-Stage Ressnet classification
                # embeddings = resnet(faces)
                #print(embeddings)
                
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Print time (inference + NMS)
                print('%sDone. (%.3fs)' % (s, time.time() - t))

                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        with open(save_path + '.txt', 'a') as file:
                            file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(c)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='models/yolov3-tiny-1cls.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/face.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='models/yolov3-tiny-1cls-20191229.pt', help='path to weights file')
    parser.add_argument('--source', type=str, default='0', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--conf_thres', type=float, default=0.28, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.4, help='IOU threshold for NMS')
    parser.add_argument('--view-img', action='store_true', help='display results')
    opt = parser.parse_args()

    with torch.no_grad():
        detect()

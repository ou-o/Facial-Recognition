import argparse

import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import test  # import test.py to get mAP after each epoch
from models import *
from utils.datasets import *
from utils.utils import *

wdir = 'weights' + os.sep  # weights dir
last = wdir + 'last.pt'
best = wdir + 'best.pt'
results_file = 'results.txt'

# Hyperparameters (results68: 59.2 mAP@0.5 yolov3-spp-416) https://github.com/ultralytics/yolov3/issues/310
hyp = {
    'giou': 3.54,  # giou loss gain
    'cls': 37.4,  # cls loss gain
    'cls_pw': 1.0,  # cls BCELoss positive_weight
    'obj': 49.5,  # obj loss gain (*=img_size/320 if img_size != 320)
    'obj_pw': 1.0,  # obj BCELoss positive_weight
    'iou_t': 0.225,  # iou training threshold
    'lr0': 0.00002,  # initial learning rate (SGD=1E-3, Adam=9E-5)
    'lrf': -4.,  # final LambdaLR learning rate = lr0 * (10 ** lrf)
    'momentum': 0.9,  # SGD momentum
    'weight_decay': 0.000484,  # optimizer weight decay
    'fl_gamma': 0.5,  # focal loss gamma
    'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
    'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
    'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
    'degrees': 1.98,  # image rotation (+/- deg)
    'translate': 0.05,  # image translation (+/- fraction)
    'scale': 0.05,  # image scale (+/- gain)
    'shear': 0.641}  # image shear (+/- deg)

def train():
    cfg = opt.cfg
    data = opt.data
    img_size = opt.img_size
    epochs = 1 if opt.prebias else opt.epochs  # 500200 batches at bs 64, 117263 images = 273 epochs
    batch_size = opt.batch_size
    accumulate = opt.accumulate  # effective bs = batch_size * accumulate = 16 * 4 = 64
    weights = opt.weights  # initial training weights

    # Initialize
    init_seeds()

    # Configure run
    data_dict = parse_data_cfg(data)
    train_path = data_dict['train']
    test_path = data_dict['valid']
    nc = int(data_dict['classes'])  # number of classes

    # Remove previous results
    for f in glob.glob('*_batch*.jpg') + glob.glob(results_file):
        os.remove(f)

    # Initialize model
    model = Darknet(cfg, arc=opt.arc).to(device)

    # Optimizer
    pg0, pg1 = [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        if 'Conv2d.weight' in k:
            pg1 += [v]  # parameter group 1 (apply weight_decay)
        else:
            pg0 += [v]  # parameter group 0

    optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    del pg0, pg1

    # https://github.com/alphadl/lookahead.pytorch
    # optimizer = torch_utils.Lookahead(optimizer, k=5, alpha=0.5)

    cutoff = -1  # backbone reaches to cutoff layer
    start_epoch = 0
    best_fitness = float('inf')
    if weights.endswith('.pt'):  # pytorch format
        # possible weights are '*.pt', 'yolov3-spp.pt', 'yolov3-tiny.pt' etc.
        chkpt = torch.load(weights, map_location=device)

        # load model
        try:
            chkpt['model'] = {k: v for k, v in chkpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(chkpt['model'], strict=False)
        except KeyError as e:
            s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " \
                "See https://github.com/ultralytics/yolov3/issues/657" % (opt.weights, opt.cfg, opt.weights)
            raise KeyError(s) from e

        # # load optimizer
        if chkpt['optimizer'] is not None:
            optimizer.load_state_dict(chkpt['optimizer'])
            best_fitness = chkpt['best_fitness']

        # load results
        if chkpt.get('training_results') is not None:
            with open(results_file, 'w') as file:
                file.write(chkpt['training_results'])  # write results.txt

        start_epoch = chkpt['epoch'] + 1
        del chkpt

    # elif len(weights) > 0:  # darknet format
    #     # possible weights are '*.weights', 'yolov3-tiny.conv.15',  'darknet53.conv.74' etc.
    #     cutoff = load_darknet_weights(model, weights)

    # if opt.transfer or opt.prebias:  # transfer learning edge (yolo) layers
    #     nf = int(model.module_defs[model.yolo_layers[0] - 1]['filters'])  # yolo layer size (i.e. 255)

    #     if opt.prebias:
    #         for p in optimizer.param_groups:
    #             # lower param count allows more aggressive training settings: i.e. SGD ~0.1 lr0, ~0.9 momentum
    #             p['lr'] *= 100  # lr gain
    #             if p.get('momentum') is not None:  # for SGD but not Adam
    #                 p['momentum'] *= 0.9

    #     for p in model.parameters():
    #         if opt.prebias and p.numel() == nf:  # train (yolo biases)
    #             p.requires_grad = True
    #         elif opt.transfer and p.shape[0] == nf:  # train (yolo biases+weights)
    #             p.requires_grad = True
            # else:  # freeze layer
            #     p.requires_grad = False

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(opt.epochs * x) for x in [0.8, 0.9]], gamma=0.1)
    scheduler.last_epoch = start_epoch - 1

    # Initialize distributed training
    if device.type != 'cpu' and torch.cuda.device_count() > 1:
        dist.init_process_group(backend='nccl',  # 'distributed backend'
                                init_method='tcp://127.0.0.1:9999',  # distributed training init method
                                world_size=1,  # number of nodes for distributed training
                                rank=0)  # distributed training node rank
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        model.yolo_layers = model.module.yolo_layers  # move yolo layer indices to top level

    # Dataset
    dataset = LoadImagesAndLabels(train_path, img_size, batch_size,
                                  augment=True,
                                  hyp=hyp,  # augmentation hyperparameters
                                  rect=opt.rect,  # rectangular training
                                  image_weights=opt.img_weights,
                                  cache_labels=epochs > 10,
                                  cache_images=opt.cache_images and not opt.prebias)

    # Dataloader
    batch_size = min(batch_size, len(dataset))
    nw = 8 # min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             shuffle=not opt.rect,  # Shuffle=True unless rectangular training is used
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    # Test Dataloader
    if not opt.prebias:
        testloader = torch.utils.data.DataLoader(LoadImagesAndLabels(test_path, opt.img_size, batch_size * 2,
                                                                     hyp=hyp,
                                                                     rect=True,
                                                                     cache_labels=True,
                                                                     cache_images=opt.cache_images),
                                                 batch_size=batch_size * 2,
                                                 num_workers=nw,
                                                 pin_memory=True,
                                                 collate_fn=dataset.collate_fn)

    # Start training
    nb = len(dataloader)
    model.nc = nc  # attach number of classes to model
    model.arc = opt.arc  # attach yolo architecture
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
    maps = np.zeros(nc)  # mAP per class
    # torch.autograd.set_detect_anomaly(True)
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    t0 = time.time()
    torch_utils.model_info(model, report='summary')  # 'full' or 'summary'
    print('Using %g dataloader workers' % nw)
    print('Starting %s for %g epochs...' % ('prebias' if opt.prebias else 'training', epochs))
    print('Start epoch %d    Totoal epochs %d' % (start_epoch, epochs))
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()
        print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))

        # Freeze backbone at epoch 0, unfreeze at epoch 1 (optional)
        freeze_backbone = False
        if freeze_backbone and epoch < 2:
            for name, p in model.named_parameters():
                # print(name,p.requires_grad)
                if int(name.split('.')[1]) < cutoff:  # if layer < 75
                    p.requires_grad = False if epoch == 0 else True
        
        # for name, p in model.named_parameters():
                # print(name,p.requires_grad)


        # Update image weights (optional)
        if dataset.image_weights:
            w = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
            image_weights = labels_to_image_weights(dataset.labels, nc=nc, class_weights=w)
            dataset.indices = random.choices(range(dataset.n), weights=image_weights, k=dataset.n)  # rand weighted idx

        mloss = torch.zeros(4).to(device)  # mean losses
        pbar = tqdm(enumerate(dataloader), total=nb)  # progress bar
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
            targets = targets.to(device)

            # Plot images with bounding boxes
            if ni == 0:
                fname = 'train_batch%g.jpg' % i
                plot_images(imgs=imgs, targets=targets, paths=paths, fname=fname)
                if tb_writer:
                    tb_writer.add_image(fname, cv2.imread(fname)[:, :, ::-1], dataformats='HWC')

            # Hyperparameter burn-in
            # n_burn = nb - 1  # min(nb // 5 + 1, 1000)  # number of burn-in batches
            # if ni <= n_burn:
            #     for m in model.named_modules():
            #         if m[0].endswith('BatchNorm2d'):
            #             m[1].momentum = 1 - i / n_burn * 0.99  # BatchNorm2d momentum falls from 1 - 0.01
            #     g = (i / n_burn) ** 4  # gain rises from 0 - 1
            #     for x in optimizer.param_groups:
            #         x['lr'] = hyp['lr0'] * g
            #         x['weight_decay'] = hyp['weight_decay'] * g

            # Run model
            pred = model(imgs)

            # Compute loss
            loss, loss_items = compute_loss(pred, targets, model)
            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss_items)
                return results

            # Scale loss by nominal batch_size of 64
            loss *= batch_size / 64

            # Compute gradient
            loss.backward()

            # Accumulate gradient for x batches before optimizing
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Print batch results
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0  # (GB)
            s = ('%10s' * 2 + '%10.3g' * 6) % (
                '%g/%g' % (epoch, epochs - 1), '%.3gG' % mem, *mloss, len(targets), img_size)
            pbar.set_description(s)

            # end batch ------------------------------------------------------------------------------------------------

        # Update scheduler
        scheduler.step()

        # Process epoch results
        final_epoch = epoch + 1 == epochs
        # if not opt.notest or final_epoch:  # Calculate mAP
        #     results, maps = test.test(cfg,
        #                               data,
        #                               batch_size=batch_size * 2,
        #                               img_size=opt.img_size,
        #                               model=model,
        #                               conf_thres=0.001 if final_epoch else 0.1,  # 0.1 for speed
        #                               dataloader=testloader)

        # Write epoch results
        with open(results_file, 'a') as f:
            f.write(s + '%10.3g' * 7 % results + '\n')

        # Write Tensorboard results
        if tb_writer:
            x = list(mloss) + list(results)
            titles = ['GIoU', 'Objectness', 'Classification', 'Train loss',
                      'Precision', 'Recall', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification']
            for xi, title in zip(x, titles):
                tb_writer.add_scalar(title, xi, epoch)

        # Update best mAP
        fitness = sum(results[4:])  # total loss
        if fitness < best_fitness:
            best_fitness = fitness

        # Save training results
        with open(results_file, 'r') as f:
            # Create checkpoint
            chkpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'training_results': f.read(),
                        'model': model.module.state_dict() if type(
                            model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                        'optimizer': None if final_epoch else optimizer.state_dict()}

        # Save last checkpoint
        torch.save(chkpt, last)

        # Save best checkpoint
        if best_fitness == fitness:
            torch.save(chkpt, best)

        # Save backup every 10 epochs (optional)
        if epoch > 0 and epoch % 10 == 0:
            torch.save(chkpt, wdir + 'backup%g.pt' % epoch)

        # Delete checkpoint
        del chkpt

        # end epoch ----------------------------------------------------------------------------------------------------

    # end training
    if len(opt.name) and not opt.prebias:
        fresults, flast, fbest = 'results%s.txt' % opt.name, 'last%s.pt' % opt.name, 'best%s.pt' % opt.name
        os.rename('results.txt', fresults)
        os.rename(wdir + 'last.pt', wdir + flast) if os.path.exists(wdir + 'last.pt') else None
        os.rename(wdir + 'best.pt', wdir + fbest) if os.path.exists(wdir + 'best.pt') else None

    plot_results()  # save as results.png
    print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
    dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()

    return results


def prebias():
    # trains output bias layers for 1 epoch and creates new backbone
    if opt.prebias:
        a = opt.img_weights  # save settings
        opt.img_weights = False  # disable settings

        train()  # transfer-learn yolo biases for 1 epoch
        create_backbone(last)  # saved results as backbone.pt

        opt.weights = wdir + 'backbone.pt'  # assign backbone
        opt.prebias = False  # disable prebias
        opt.img_weights = a  # reset settings


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300) 
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--accumulate', type=int, default=8, help='batches to accumulate before optimizing')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-tiny-1cls.cfg', help='*.cfg path')
    parser.add_argument('--data', type=str, default='data/wider.data', help='*.data path')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
    parser.add_argument('--transfer', action='store_true', help='transfer learning')
    parser.add_argument('--img-weights', action='store_true', help='select training images by weight')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--weights', type=str, default='weights/best.pt', help='initial weights')
    parser.add_argument('--arc', type=str, default='default', help='yolo architecture')
    parser.add_argument('--prebias', action='store_true', help='transfer-learn yolo biases prior to training')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--var', type=float, help='debug variable')
    opt = parser.parse_args()
    opt.weights = last if opt.resume else opt.weights
    print(opt)
    device = torch_utils.select_device(opt.device, batch_size=opt.batch_size)

    tb_writer = None
    try:
        # Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/
        from torch.utils.tensorboard import SummaryWriter

        tb_writer = SummaryWriter()
    except:
        pass

    prebias()  # optional
    train()  # train normally


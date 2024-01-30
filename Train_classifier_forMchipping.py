import os
import time
import copy
import random
import shutil
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn.functional as F
import cv2
import cv2 as cv
from utils.torch_dataset import ImageFolderDatasetWithValid, letterbox, image_to_model_input
from utils.torch_device import get_device, gpu_id
from utils.general import one_cycle, increment_path, EMA
from utils.export import export_onnx
from utils.torch_debug import show_image, conv_visualization
from utils.loss import label_smoothing
from cfg_forMchipping import ConfigData, HypScratch
from cnn_model import initialize_efficientnet_b4
from cnn_model import initialize_resnet50
from cnn_model import initialize_resnet34
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
from utils.scan_files import scan_files_subfolder
# ignore all future warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

def parse_opt(known=False):
    c = ConfigData()
    hyp = HypScratch()
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='train or find_label_noise')
    parser.add_argument('--img-folder', type=str, default=c.img_folder_root)
    parser.add_argument('--output-root', type=str, default=c.output_root, help='save to output-root/output-name')
    parser.add_argument('--output-name', type=str, default=c.output_name, help='save to output-root/output-name')
    parser.add_argument('--weights', type=str, default=c.pretrained_weights, help='initial weights path')   
    parser.add_argument('--valid-keep', type=float, default=c.valid_keep_ratio)
    parser.add_argument('--img-newsize', type=int, default=c.img_newsize, help='train, val image size (pixels)')
    parser.add_argument('--device-mem', default=c.device_memory_ratio, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=c.workers, help='maximum number of dataloader workers')
    parser.add_argument('--epochs', type=int, default=c.epochs)
    parser.add_argument('--early-stop', type=int, default=c.early_stop)
    parser.add_argument('--batch-size', type=int, default=c.batch_size)
    parser.add_argument('--num-classes', type=int, default=c.num_classes)
    parser.add_argument('--over-sampling-thresh', type=int, default=c.over_sampling_thresh)
    parser.add_argument('--over-sampling-scale', type=int, default=c.over_sampling_scale)
    parser.add_argument('--lr', type=float, default=hyp.lr)
    parser.add_argument('--lrf', type=float, default=hyp.lrf)
    parser.add_argument('--momentum', type=float, default=hyp.momentum)
    parser.add_argument('--weight_decay', type=float, default=hyp.weight_decay)
    parser.add_argument('--warmup_epochs', type=int, default=hyp.warmup_epochs)
    parser.add_argument('--warmup_momentum', type=float, default=hyp.warmup_momentum)
    parser.add_argument('--warmup_bias_lr', type=float, default=hyp.warmup_bias_lr)
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    for o in opt._get_kwargs():
        print(o)
    print('')
    return opt


def save_train_curve(train_loss_history, val_loss_history, val_f1_history, output_folder):
    # plot history
    fig, ax = plt.subplots(1, 3)
    ax[0].plot(train_loss_history)
    ax[1].plot(val_loss_history)
    ax[2].plot(val_f1_history)
    ax[0].set_title('train_loss_history')
    ax[1].set_title('val_loss_history')
    ax[2].set_title('val_f1_history')
    plt.tight_layout(pad=1)
    plt.savefig(f'{output_folder}/histroy.png')
    # plt.show()
    plt.close()


def train(opt):
    
    # model
    model = initialize_resnet50(opt.num_classes, opt.weights)
    model = model.to(get_device())

    # dataset
    ds = ImageFolderDatasetWithValid(
        opt.img_folder,
        opt.img_newsize,
        opt.valid_keep,
        over_sampling_thresh=opt.over_sampling_thresh,
        over_sampling_scale=opt.over_sampling_scale)

    # dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset=ds, 
        batch_size=opt.batch_size, 
        num_workers=opt.workers, 
        pin_memory=True, 
        shuffle=True)
    dataloader.dataset.set_train(True)

    # optimizer parameter groups
    g0, g1, g2 = [], [], []
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, torch.nn.Parameter): # bias
            g2.append(v.bias)
        if isinstance(v, torch.nn.BatchNorm2d): # weight (no decay)
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, torch.nn.Parameter): # weight (with decay)
            g1.append(v.weight)

    optimizer = torch.optim.SGD(g0, lr=opt.lr, momentum=opt.momentum, nesterov=True)        
    optimizer.add_param_group({'params': g1, 'weight_decay': opt.weight_decay})  # add g1 with weight_decay
    optimizer.add_param_group({'params': g2})  # add g2 (biases)

    # compute class weights and build loss function
    _, label_idx, label_name = ds.get_label_info()
    weights = compute_class_weight('balanced', label_idx, ds.get_labels(use_train=True))
    class_weights = torch.FloatTensor(weights).to(get_device())
    # loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    loss_fn = torch.nn.BCEWithLogitsLoss(weight=class_weights)
    for i, name in enumerate(label_name):
        print(f'[{name}] class_weights = {class_weights[i]}')

    # initial ema
    ema = EMA(model, 0.999)
    ema.register()

    #--------------------------------------- Train Start --------------------------------------
    since = time.time()
    cost_time = since

    num_epochs = opt.epochs
    warmup_epoch = opt.warmup_epochs
    early_stop = opt.early_stop
    img_newsize = opt.img_newsize
    bs = opt.batch_size
    output_folder = increment_path(f'{opt.output_root}\\{opt.output_name}', mkdir=True)

    train_loss_history = []
    val_loss_history = []
    val_f1_history = []
    best_model_pt = copy.deepcopy(model.state_dict())
    best_loss = 99999999.0
    best_f1_score = 0.0   
    early_stop_count = 0

    dataloader.dataset.set_train(True)
    nb = len(dataloader)  # number of batches
    nw = min(round(warmup_epoch * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    lf = one_cycle(1, opt.lrf, num_epochs)  # cosine 1->hyp['lrf']
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                dataloader.dataset.set_train(True)
                model.train()
                ema.restore()
            else:
                dataloader.dataset.set_train(False)
                model.eval()
                ema.apply_shadow()

            data_num = len(dataloader)
            running_loss = 0.0
            running_corrects = 0
            running_labels = []
            running_preds = []

            # Iterate over data.
            for index, loader in enumerate(dataloader):
                paths, imgs, inputs, labels = loader
                inputs = inputs.to(get_device(), non_blocking=True)
                labels = labels.to(get_device(), non_blocking=True)

                # zero the parameter gradients
                optimizer.zero_grad()

                # Warmup
                ni = index + nb * epoch  # number integrated batches (since train start)
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    for j, x in enumerate(optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(ni, xi, [opt.warmup_bias_lr if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [opt.warmup_momentum, opt.momentum])
                # print('optimizer = ', optimizer)

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs, conv_outputs = model(inputs)
                    # fc = list(model.fc.modules())[1]
                    # conv_visualization(imgs, conv_outputs, outputs, fc.weight, fc.bias)

                    _, preds = torch.max(outputs, 1)        

                    label_one_hot = F.one_hot(labels, 10).float() ######################################## change to num class
                    label_one_hot = label_smoothing(label_one_hot, opt.num_classes, epsilon=0.1)
                    loss = loss_fn(outputs, label_one_hot)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        ema.update()

                # statistics
                batch_correct = torch.sum(preds == labels.data)
                batch_acc = batch_correct / inputs.size(0)
                batch_f1 = f1_score(labels.data.to('cpu'), preds.to('cpu'), average='weighted')
                running_loss += loss.item() * inputs.size(0)
                running_corrects += batch_correct
                running_labels += labels.data.to('cpu')
                running_preds += preds.to('cpu')

                # print message
                if phase == 'train':  
                    print('[Epoch {:03d}, Step {:04d}/{:04d}] Loss: {:.4f} Acc: {:.4f} F1: {:.4f}'.format(
                        epoch, index, data_num, loss.item(), batch_acc, batch_f1))
   
            # empty cuda cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            epoch_loss = 0.0
            epoch_acc = 0.0
            epoch_f1 = 0.0
            if data_num > 0:
                epoch_loss = running_loss / len(dataloader.dataset)
                epoch_acc = running_corrects.double() / len(dataloader.dataset)
                epoch_f1 = f1_score(running_labels, running_preds, average='weighted')

                # print message
                print('{} Loss: {:.4f} Acc: {:.4f} F1: {:.4f} CostTime: {}'.format(
                    phase, epoch_loss, epoch_acc, epoch_f1, time.time() - cost_time))
            cost_time = time.time()

            # save history loss
            if phase == 'train':
                train_loss_history.append(epoch_loss)
            elif phase == 'val':
                val_loss_history.append(epoch_loss)
                val_f1_history.append(epoch_f1)
                # save history
                save_train_curve(train_loss_history, val_loss_history, val_f1_history, output_folder)

            # save best f1_score
            if phase == 'val' and data_num > 0 and epoch > warmup_epoch:
                if (epoch_f1 > best_f1_score) or (epoch_f1 == best_f1_score and epoch_loss < best_loss):
                    early_stop_count = 0
                    best_loss = epoch_loss
                    best_f1_score = epoch_f1

                    # save best.txt
                    save_path = f'{output_folder}\\best.txt'
                    with open(save_path, 'w') as f:
                        f.write(f'[epoch {epoch}] best_val_loss = {best_loss}, best_val_f1_score = {best_f1_score}\n')

                    # save best.pt
                    best_model_pt = copy.deepcopy(model.state_dict())
                    save_path = f'{output_folder}\\best.pt'
                    torch.save(best_model_pt, save_path)

                    # save best.onnx
                    export_onnx(model, [1, 3, img_newsize, img_newsize], output_folder)

        # update lr_scheduler
        scheduler.step()
        # print('scheduler.get_lr() = ', scheduler.get_lr(), scheduler.get_last_lr())

        # early stop
        early_stop_count += 1
        if early_stop_count >= early_stop:
            print(f'[epoch {epoch}] early stop!')
            break
        print()

    #--------------------------------------- Train Done --------------------------------------
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # save history
    save_train_curve(train_loss_history, val_loss_history, val_f1_history, output_folder)


def test(opt):
    opt.img_folder = r''
    opt.weights = r''
    model = initialize_resnet50(opt.num_classes, opt.weights)
    model = model.to(get_device())
    model.eval()
    # dataset
    img_newsize = opt.img_newsize
    img_folder_root = opt.img_folder
    files = scan_files_subfolder(img_folder_root, ['jpg','jpeg', 'bmp', 'png'])
    random.shuffle(files)
    # files = files[:200]
    n = len(files)
    with torch.no_grad():
        for index, file in enumerate(files):
            
            # load
            img = cv2.imdecode(np.fromfile(file, dtype=np.uint8), -1)
            # img = center_crop(img, (512, 512))
            img, ratio, pad = letterbox(img, (img_newsize, img_newsize))

            # to tensor
            inputs = image_to_model_input(img, True)
            inputs = inputs.to(get_device())

            # # inference
            outputs = []
            outputs, conv_outputs = model(inputs)           
            fc = list(model.fc.modules())[1]
            pred = int(torch.argmax(outputs[0]))
            print(f'[{index}/{n}] inference outputs = {outputs}, pred = {pred}, path = {file}')   

def main(opt):
    # gpu memory limit
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        torch.cuda.set_per_process_memory_fraction(opt.device_mem, gpu_id)
        torch.cuda.empty_cache()
        print('torch.cuda.current_device() = ', torch.cuda.current_device())
    if opt.mode == 'train':
        train(opt)
    if opt.mode == 'test':
        test(opt)

if __name__ == '__main__':
    opt = parse_opt(True)
    main(opt)
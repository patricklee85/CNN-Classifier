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

from cfg import ConfigData, HypScratch
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
    parser.add_argument('--mode', type=str, default='test', help='train or find_label_noise')
    parser.add_argument('--img-folder', type=str, default=c.img_folder_root)
    parser.add_argument('--output-root', type=str, default=c.output_root, help='save to output-root/output-name')
    parser.add_argument('--output-name', type=str, default=c.output_name, help='save to output-root/output-name')
    # parser.add_argument('--cfg', type=str, default=c.pretrained_cfg, help='model.yaml path')
    parser.add_argument('--weights', type=str, default=c.pretrained_weights, help='initial weights path')   
    parser.add_argument('--valid-keep', type=float, default=c.valid_keep_ratio)
    parser.add_argument('--img-newsize', type=int, default=c.img_newsize, help='train, val image size (pixels)')
    # parser.add_argument('--device', default=c.device, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--device-mem', default=c.device_memory_ratio, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=c.workers, help='maximum number of dataloader workers')
    parser.add_argument('--epochs', type=int, default=c.epochs)
    parser.add_argument('--early-stop', type=int, default=c.early_stop)
    parser.add_argument('--batch-size', type=int, default=c.batch_size)
    parser.add_argument('--num-classes', type=int, default=c.num_classes)
    parser.add_argument('--over-sampling-thresh', type=int, default=c.over_sampling_thresh)
    parser.add_argument('--over-sampling-scale', type=int, default=c.over_sampling_scale)
    # parser.add_argument('--use-adam', type=bool, default=c.use_adam, help='use torch.optim.Adam() optimizer')
    # parser.add_argument('--use-finetune', type=bool, default=c.use_finetune)
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
   
def softmax(x_input):
    row_max = np.max(x_input)
    x_input = x_input - row_max
    x_exp = np.exp(x_input)
    x_sum = np.sum(x_exp)
    s = x_exp/x_sum
    return s_output
    

def test(opt):
    opt.img_folder = r'D:\BackendAOI\Photo\Offline\Test\20240129\4HW3-2802-TR1C\AO3001F\GT_ADC_FF\Bin8_Pic1'
    opt.weights = r'D:\BackendAOI\Python\cnn\save_model\1.A73A_CNN_model\4HW3-2802-TR1C_Fm_V3\2class\best.pt'
    model = initialize_resnet50(opt.num_classes, opt.weights)
    model = model.to(get_device())
    model.eval()

    # dataset
    img_newsize = opt.img_newsize
    img_folder_root = opt.img_folder
    files = scan_files_subfolder(img_folder_root, ['jpg','jpeg','bmp','png'])
    random.shuffle(files)
    # files = files[:200]
    # add output report
    os.makedirs(opt.img_folder+'\\2class_output_report', exist_ok=True)
    txt_file = open(opt.img_folder+'\\2class_output_report\\results.txt','w')
    os.makedirs(opt.img_folder+'\\2class_output_report\\pass_output', exist_ok=True)
    os.makedirs(opt.img_folder+'\\2class_output_report\\fail_output', exist_ok=True)

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
            #pred_softmax = softmax(conv_outputs)
            print(f'[{index}/{n}] inference outputs = {outputs}, pred = {pred}, path = {file}') 
            #return  
            txt_file.write("class pass score = "+(str(outputs).split(", ")[0]).split('([[')[1]+", class fail score = "+(str(outputs).split(", ")[1]).split(']]')[0]+"\n")    
            txt_file.write(f'[{index}/{n}] path = {file} , pred = {pred}'+"\n")
            if f'{pred}' == '0':
                #if float((str(outputs).split(", ")[0]).split('([[')[1]) > 2.5: 
                    pass_img = cv2.imread(f'{file}')
                    cv2.imwrite(opt.img_folder+'/2class_output_report/pass_output/'+'Pass'+(str(outputs).split(", ")[0]).split('([[')[1]+"_"+(str(outputs).split(", ")[1]).split(']]')[0]+".jpg",pass_img)
                #else:
                    #nornal_img = cv2.imread(f'{file}')
                    #cv2.imwrite(opt.img_folder+'/output_report/blur_to_normal/'+(str(outputs).split(", ")[0]).split('([[')[1]+"_"+(str(outputs).split(", ")[1]).split(']]')[0]+".jpg",nornal_img)
            if f'{pred}' == '1':
                fail_img = cv2.imread(f'{file}')
                cv2.imwrite(opt.img_folder+'/2class_output_report/fail_output/'+'Ng'+(str(outputs).split(", ")[0]).split('([[')[1]+"_"+(str(outputs).split(", ")[1]).split(']]')[0]+".jpg",fail_img)

def main(opt):
    # gpu memory limit
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        torch.cuda.set_per_process_memory_fraction(opt.device_mem, gpu_id)
        torch.cuda.empty_cache()
        print('torch.cuda.current_device() = ', torch.cuda.current_device())
    if opt.mode == 'test':
        test(opt)

    
if __name__ == '__main__':
    opt = parse_opt(True)
    main(opt)
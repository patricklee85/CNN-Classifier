
import os
import time
import copy
import random
import shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from utils.torch_dataset import ImageFolderDataset, ImageFolderDatasetWithValid
from utils.torch_device import get_device, gpu_id
from utils.general import one_cycle, increment_path
from utils.export import export_onnx
from utils.scan_files import scan_files_subfolder
from utils.torch_debug import show_image

from cfg import ConfigData
from cnn_model import initialize_resnet50
from cleanlab.pruning import get_noise_indices

# ignore all future warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


def find_label_noise(opt):
    cfg = ConfigData()

    cfg.img_debug = False
    cfg.img_folder = r'D:\BackendAOI\data\classification\4HW3\1_chipping'
    cfg.output_folder = r'D:\BackendAOI\data\classification\4HW3\1_chipping_Unknown'
    cfg.weights = r'D:\BackendAOI\py\cnn\save_model\4HW3_chipping_2class\classifier\best.pt'

    model = initialize_resnet50(cfg.classes_num, cfg.weights)
    model = model.to(get_device())
    model.eval()

    # dataset
    img_newsize = cfg.img_newsize
    img_folder_root = cfg.img_folder
    ds = ImageFolderDataset(
        image_folder_root=img_folder_root,
        image_newsize=img_newsize)

    # dataloader
    data_loader = torch.utils.data.DataLoader(
        dataset=ds, 
        batch_size=cfg.batch_size, 
        num_workers=cfg.workers, 
        pin_memory=True, 
        shuffle=False)

    # output
    img_folder_unlabeled = cfg.output_folder
    os.makedirs(img_folder_unlabeled, exist_ok=True)

    # run
    print('find_label_noise, start inference...')
    # if os.path.exists('npy/psx.npy'):
    if False:
        psx = np.load('npy/psx.npy')
    else:
        output_list = []
        n = len(data_loader)
        with torch.no_grad():
            for index, loader in enumerate(data_loader):
                paths, imgs, inputs, labels = loader
                inputs = inputs.to(get_device(), non_blocking=True)
                # labels = labels.to(get_divece(), non_blocking=True)

                # inference
                outputs, _ = model(inputs)
                output_list.append(outputs)
                print(f'[{index}/{n}] inference done')
                # torch_debug.show_image(img)

        # Convert to probabilities and return the numpy array of shape N x K
        output_list = torch.cat(output_list, dim=0)
        output_list = output_list.cpu().detach().numpy()

        # save
        psx = np.exp(output_list)
        os.makedirs('npy', exist_ok=True)
        np.save('npy/psx.npy', psx)
    print('find_label_noise, inference done')

    # get noise label
    s = ds.get_labels()
    ordered_label_errors = get_noise_indices(
        s=s,
        psx=psx,
        n_jobs=4,
        num_to_remove_per_class=100,
        prune_method='prune_by_class',
        # prune_method='prune_by_noise_rate',
        sorted_index_method='normalized_margin', # Orders label errors
    )
    print('find_label_noise, get_noise_indices done')

    images = np.array(ds.get_images())
    labels = np.array(ds.get_labels())
    confs = psx[ordered_label_errors]
    images = images[ordered_label_errors]
    labels = labels[ordered_label_errors]
    print(f'move label_noise to ({img_folder_unlabeled})')
    for im, lbl in zip(images, labels):
        im_new = im.replace(img_folder_root, img_folder_unlabeled)
        os.makedirs(os.path.dirname(im_new), exist_ok=True)
        shutil.move(im, im_new)
        print(f'move {im} to ({im_new})')
 

def main():
    # gpu memory limit
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        torch.cuda.set_per_process_memory_fraction(0.99, gpu_id)
        torch.cuda.empty_cache()
        print('torch.cuda.current_device() = ', torch.cuda.current_device())

    find_label_noise()

    
if __name__ == '__main__':
    main()
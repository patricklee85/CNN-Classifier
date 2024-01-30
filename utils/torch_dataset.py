
import os
import cv2
import torch
import random
import numpy as np

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from utils.scan_files import scan_dirs_folder, scan_files_subfolder
from utils.augmentations import center_crop, letterbox, augment_hsv, random_perspective, Albumentations


def image_to_model_input(img, extend_batch_dim=False): 
    inputs = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    inputs = np.ascontiguousarray(inputs)
    inputs = torch.from_numpy(inputs)
    inputs = inputs.float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0   
    if extend_batch_dim:
        inputs = torch.unsqueeze(inputs, 0)
    return inputs

def image_augment(img):

    # random_perspective
    img, _ = random_perspective(
        img, 
        degrees=5,
        translate=0.01,
        scale=0.1,
        shear=3,
        perspective=0.0003)

    # Albumentations
    albumentations = Albumentations()
    img = albumentations(img)

    # HSV color-space
    augment_hsv(img, hgain=0.015, sgain=0.1, vgain=0.1)

    # Flip up-down
    if random.random() < 0.5:
        img = np.flipud(img)

    # Flip left-right
    if random.random() < 0.5:
        img = np.fliplr(img)

    # # debug    
    # import utils.torch_debug
    # utils.torch_debug.show_image(img)

    return img


class ImageFolderDatasetWithValid(torch.utils.data.Dataset):
    def __init__(
        self, 
        image_folder_root,
        image_newsize,
        valid_keep_ratio,
        over_sampling_thresh=-1,
        over_sampling_scale=1,
        ignore_folder_name='Unlabeled'):

        self.image_newsize = image_newsize
        self.use_train = True
        self.train_list = []
        self.valid_list = []    
        self.label_names = []

        dirs = scan_dirs_folder(image_folder_root)
        dirs = [d for d in dirs if os.path.basename(d) != ignore_folder_name]
        for label, d in enumerate(dirs):
            # label name
            self.label_names.append(os.path.basename(d))
            # image files
            files = scan_files_subfolder(d, ['jpg','jpeg', 'bmp', 'png'])
            if len(files) < over_sampling_thresh:
                files = files * over_sampling_scale
            # shuffle
            random.shuffle(files)
            # spilt
            path_train, path_valid = [], []
            if valid_keep_ratio == 0:
                path_train, path_valid = files, []
            elif valid_keep_ratio == 1:
                path_train, path_valid = [], files
            else:
                path_train, path_valid = train_test_split(
                    files, 
                    test_size=valid_keep_ratio, 
                    random_state=42)
            # append
            [self.train_list.append([p, label]) for p in tqdm(path_train)]
            [self.valid_list.append([p, label]) for p in tqdm(path_valid)]
        # write label.txt
        self.label_num = len(self.label_names)
        with open(f'{image_folder_root}\\label.txt', 'w') as f:
            [f.write(f'{i} {c}\n') for i, c in enumerate(self.label_names)]

        print(f'ImageFolderDatasetWithValid, train data count = {len(self.train_list)}')
        print(f'ImageFolderDatasetWithValid, valid data count = {len(self.valid_list)}')
        print(f'ImageFolderDatasetWithValid, number of label = {self.label_num}')
        print(f'ImageFolderDatasetWithValid, label = {self.label_names}')
        
    def __getitem__(self, index):
        # load
        path, label = self.train_list[index] if self.use_train else self.valid_list[index]
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
        # img = center_crop(img, (800, 800))

        # augmentation
        if self.use_train:            
            img = image_augment(img)

        # letterbox
        sz = self.image_newsize
        img, ratio, pad = letterbox(img, (sz, sz))

        # # debug    
        # import utils.torch_debug
        # utils.torch_debug.show_image(img)

        # to tensor
        tensor = image_to_model_input(img)
        return (path, img.copy(), tensor, label)

    def __len__(self):
        n = len(self.train_list) if self.use_train else len(self.valid_list)
        return n

    def set_train(self, use_train):
        self.use_train = use_train

    def get_label_info(self):
        return len(self.label_names), list(range(len(self.label_names))), self.label_names

    def get_labels(self, use_train):
        if use_train:
            labels = [item[1] for item in self.train_list]
            return labels
        else:
            labels = [item[1] for item in self.valid_list]
            return labels

    def get_images(self, use_train):
        if use_train:
            images = [item[0] for item in self.train_list]
            return images
        else:
            images = [item[0] for item in self.valid_list]
            return images


class ImageFolderDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        image_folder_root,
        image_newsize,
        use_aug=False,
        over_sampling_thresh=-1,
        over_sampling_scale=1,
        ignore_folder_name='Unlabeled'):

        self.image_newsize = image_newsize
        self.image_list = [] 
        self.label_names = []
        self.use_aug = use_aug

        dirs = scan_dirs_folder(image_folder_root)
        dirs = [d for d in dirs if os.path.basename(d) != ignore_folder_name]
        for label, d in enumerate(dirs):
            # label name
            self.label_names.append(os.path.basename(d))
            # image files
            files = scan_files_subfolder(d, ['jpg','jpeg', 'bmp', 'png'])
            if len(files) < over_sampling_thresh:
                files = files * over_sampling_scale
            # shuffle
            random.shuffle(files)
            # append
            [self.image_list.append([p, label]) for p in tqdm(files)]
        # write label.txt
        self.label_num = len(self.label_names)
        with open(f'{image_folder_root}\\label.txt', 'w') as f:
            [f.write(f'{i} {c}\n') for i, c in enumerate(self.label_names)]

        print(f'ImageFolderDataset, image data count = {len(self.image_list)}')
        print(f'ImageFolderDataset, number of label = {self.label_num}')
        print(f'ImageFolderDataset, label = {self.label_names}')
        
    def __getitem__(self, index):
        # load
        path, label = self.train_list[index] if self.use_train else self.valid_list[index]
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)

        # augmentation
        if self.use_aug:            
            img = image_augment(img)

        # letterbox
        sz = self.image_newsize
        img, ratio, pad = letterbox(img, (sz, sz))

        # # debug    
        # import utils.torch_debug
        # utils.torch_debug.show_image(img)

        # to tensor
        tensor = image_to_model_input(img)
        return (path, img.copy(), tensor, label)

    def __len__(self):
        n = len(self.image_list)
        return n

    def get_label_info(self):
        return len(self.label_names), list(range(len(self.label_names))), self.label_names

    def get_labels(self):
        labels = [item[1] for item in self.image_list]
        return labels

    def get_images(self):
        images = [item[0] for item in self.image_list]
        return images

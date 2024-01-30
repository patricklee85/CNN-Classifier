import cv2
import os
import argparse
import numpy as np
# CMD : python image_tool.py --mode ImgCenterCrop --imgPath r'D:\111_PK5533\1_chipping\1.Pass' --outPath r'D:\111_PK5533\1_chipping_crop\1.Pass' --crop_size 512

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='ImgCenterCrop', help = 'ImgCenterCrop ImgRename ImgResize ImgPadding' )
    parser.add_argument('--imgPath', type=str, default = r'D:\111_PK5533\1_chipping\1.Pass')
    parser.add_argument('--outPath', type=str, default = r'D:\111_PK5533\1_chipping_crop\1.Pass')
    parser.add_argument('--crop_size', type=int, default = 512)
    parser.add_argument('--re_size', type=int, default = 512)
    parser.add_argument('--padding_size', type=int, default = 512)
    parser.add_argument('--new', type=str, default = 'new')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    for optgrm in opt._get_kwargs():
        print(optgrm)
    return opt

#..............................................................................................................#      
  
def img_center_crop(imgPath, outPath, crop_size):
    allFileList = os.listdir(imgPath)
    for file in allFileList:
        img = cv2.imread(imgPath+'\\'+file)
        h, w, c = img.shape
        crop_w = int(crop_size)
        crop_h = int(crop_size)
        crop_x = int(w/2-crop_size/2)
        crop_y = int(h/2-crop_size/2)
        crop_img = img[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        cv2.imwrite(outPath+'\\'+file, crop_img)

def img_Rename(imgPath, outPath, new):
    allFileList = os.listdir(imgPath)
    for file in allFileList:
        img = cv2.imread(imgPath+'\\'+file)
        cv2.imwrite(outPath+'\\'+new+'_'+file, img)

def Img_Resize(imgPath, outPath, re_size):
    allFileList = os.listdir(imgPath)
    for file in allFileList:
        img = cv2.imread(imgPath+'\\'+file)
        re_img = cv2.resize(img, (re_size, re_size), interpolation=cv2.INTER_AREA)
        cv2.imwrite(outPath+'\\'+file, re_img)

def Img_Padding(imgPath, outPath, padding_size):
    allFileList = os.listdir(imgPath)
    for file in allFileList:
        img = cv2.imread(imgPath+'\\'+file)
        h, w, c = img.shape
        color = (0,0,0)
        result = np.full((padding_size,padding_size, c), color, dtype=np.uint8)
        padding_img = result[(padding_size - h) // 2:((padding_size - h) // 2)+h, (padding_size - w) // 2:((padding_size - w) // 2)+w] 
        cv2.imshow("result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite(outPath+'\\'+file, padding_img)
    
#..............................................................................................................#     
   
def Main(opt):
    if opt.mode == 'ImgCenterCrop':
       img_center_crop(opt.imgPath, opt.outPath, opt.crop_size)
    if opt.mode == 'ImgRename':
       img_Rename(opt.imgPath, opt.outPath, opt.new)
    if opt.mode == 'ImgResize':
       Img_Resize(opt.imgPath, opt.outPath, opt.re_size)
    if opt.mode == 'ImgPedding':
       Img_Padding(opt.imgPath, opt.outPath, opt.padding_size)

if __name__ == '__main__':
    opt = parse_opt(True)
    Main(opt) 
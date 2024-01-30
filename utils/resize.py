import cv2
import os
yourPath = 'D:/BackendAOI/Data/VC/DATA_2/TEST/NG/'
allFileList = os.listdir(yourPath)
for file in allFileList:
    print(file)
    img = cv2.imread("D:/BackendAOI/Data/VC/DATA_2/TEST/NG/"+file.split('.')[0]+".jpg")
    resize_img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    cv2.imwrite('D:/BackendAOI/Data/VC/DATA_2/TEST/NG/'+file.split('.')[0]+".jpg", resize_img)
    cv2.waitKey(0)

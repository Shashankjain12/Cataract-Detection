import cv2
import os

for i in os.listdir():
    img=cv2.imread(i)
    black=cv2.cvtColor(img,cv2.BGR2GRAY)
    res=cv2.resize(black,(128,128),interpolation=cv2.INTER_AREA)
    cv2.imwrite("img_"+i,res)


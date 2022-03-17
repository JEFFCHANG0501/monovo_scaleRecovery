import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
from mono_vo.visual_odometry import PinholeCamera, VisualOdometry



def main():
    cam = PinholeCamera(1241.0, 376.0, 718.8560, 718.8560, 607.1928, 185.2157)
    vo = VisualOdometry(cam)
    for img_id in range(4541):
        img = cv2.imread('/mnt/d/mono_vo/data/data_odometry_gray/dataset/sequences/00/image_1/'+str(img_id).zfill(6)+'.png', 0)
        vo.update(img, img_id)
        
        img_c = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        for i in range(vo.px_ref.shape[0]):
            cv2.circle(img_c,(int(vo.px_ref[i,0]),int(vo.px_ref[i,1])),2,(255,0,0),-1)
        
        cv2.imshow('view', img_c)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
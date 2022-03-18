import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
from mono_vo.visual_odometry import PinholeCamera, VisualOdometry
from scipy.spatial import Delaunay
from numpy.linalg import inv

def main():
    cam = PinholeCamera(1241.0, 376.0, 718.8560, 718.8560, 607.1928, 185.2157)
    vo = VisualOdometry(cam)

    image_id = 0
    begin_id = 0
    feature2d = None

    for img_id in range(4541):
        img = cv2.imread('/mnt/d/mono_vo/data/data_odometry_gray/dataset/sequences/00/image_1/'+str(img_id).zfill(6)+'.png', 0)
        vo.update(img, img_id)
        
        if image_id > begin_id:

            feature2d = vo.feature3d[:,0:2].copy()
            feature2d[:,0] = feature2d[:,0]*cam.fx/vo.feature3d[:,2] + cam.cx
            feature2d[:,1] = feature2d[:,1]*cam.fx/vo.feature3d[:,2] + cam.cy

            lower_feature_ids = feature2d[:,1] > 185
            feature2d = feature2d[lower_feature_ids,:]
            feature3d = vo.feature3d[lower_feature_ids,:]
            print (feature2d.size)
            tri = Delaunay(feature2d)
            triangle_ids = tri.simplices

            b_matrix = np.ones((3,1),np.float)
            triangles_inv = np.array([np.matrix(feature3d[triangle_id]).I for triangle_id in triangle_ids])
            normals = (triangles_inv@b_matrix).reshape(-1,3)
            normals_len = np.sqrt(np.sum(normals*normals,1)).reshape(-1,1)
            normals = normals/normals_len
            pitch_deg = np.arcsin(-normals[:,1])*180/np.pi
            valid_pitch_id = pitch_deg < -80
            valid_pitch_id_tight = pitch_deg < -85
            # print('triangle left ',np.sum(valid_pitch_id),'from',valid_pitch_id.shape[0])
            heights = (1/normals_len).reshape(-1)
            
            hieght_level = 0.9*np.median(heights[valid_pitch_id])
            valid_height_id = heights > hieght_level
            valid_id = valid_pitch_id_tight & valid_height_id
            # print('triangle left final',np.sum(valid_id),'from',valid_id.shape[0])
            valid_points_id = triangle_ids[valid_id].reshape(-1)
            
            point_selected = feature3d[(valid_points_id)]
            triangle_points = np.array(feature2d[valid_points_id], np.int32)
            triangle_points_line = triangle_points.reshape((-1,1,2))
            

            img_c = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            for i in range(triangle_points.shape[0]):
                cv2.circle(img_c,(int(triangle_points[i,0]),int(triangle_points[i,1])),2,(255,0,0),-1)
            cv2.polylines(img_c,[triangle_points_line],True,(0,255,0),1)
            cv2.fillPoly(img_c,[triangle_points_line],(0,255,0))
            cv2.imshow('view', img_c)
            cv2.waitKey(1)
        
        
        image_id += 1

if __name__ == '__main__':
    main()
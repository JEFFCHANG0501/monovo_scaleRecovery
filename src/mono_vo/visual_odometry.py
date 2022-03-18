import numpy as np 
import cv2
from numpy.linalg import inv

STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2
kMinNumFeature = 1000

lk_params = dict(winSize  = (21, 21), 
            maxLevel = 3,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

def feature_check(kp1,kp2):
    dis = np.sum(np.abs(kp1-kp2),1)
    valid = dis>=1
    return kp1[valid,:],kp2[valid,:]

def featureTracking(image_ref, image_cur, px_ref):
    kp2, state, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params)
    state = state.reshape(state.shape[0])
    kp1 = px_ref[state == 1]
    kp2 = kp2[state == 1]
    # kp1,kp2 = feature_check(kp1,kp2)
    return kp1, kp2


class PinholeCamera:
    def __init__(self, width, height, fx, fy, cx, cy, 
                k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

class VisualOdometry:
    def __init__(self, cam, annotations = None):
        self.cam = cam
        self.frame_stage = 0
        self.focal = cam.fx
        self.pp = (cam.cx, cam.cy)
        self.new_frame = None
        self.last_frame = None
        self.px_ref = None
        self.px_cur = None
        self.px_ref_selected = None
        self.px_cur_selected = None
        self.cur_R = None
        self.cur_t = None
        self.camera_matrix = np.eye(3)
        self.camera_matrix[0,0] = self.camera_matrix[1,1] = self.focal
        self.camera_matrix[0,2] = cam.cx
        self.camera_matrix[1,2] = cam.cy
        self.feature3d = None
        self.detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)

    def processFirstFrame(self):
        self.px_ref = self.detector.detect(self.new_frame)
        self.px_ref = np.array([x.pt for x in self.px_ref], dtype=np.float32)
        self.frame_stage = STAGE_SECOND_FRAME
        

    def processSecondFrame(self):
        self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref)
        
        E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, cameraMatrix = self.camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, self.cur_R, self.cur_t, mask, points_3d = cv2.recoverPose(E, self.px_cur, self.px_ref, cameraMatrix = self.camera_matrix, distanceThresh=100)
        mask_bool = np.array(mask>0).reshape(-1)
        points_3d_selected = points_3d [:,mask_bool].T
        points_3d_selected[:,0] = points_3d_selected[:,0]/points_3d_selected[:,3]
        points_3d_selected[:,1] = points_3d_selected[:,1]/points_3d_selected[:,3]
        points_3d_selected[:,2] = points_3d_selected[:,2]/points_3d_selected[:,3]
        
        self.feature3d = points_3d_selected[:,0:3]
        self.px_cur_selected = self.px_cur[mask_bool,:]
        self.px_ref_selected = self.px_ref[mask_bool,:]

        self.frame_stage = STAGE_DEFAULT_FRAME
        self.px_ref = self.px_cur

    def processFrame(self, frame_id):
        self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref)
        
        E, mask_e = cv2.findEssentialMat(self.px_cur, self.px_ref, cameraMatrix = self.camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, self.cur_R, self.cur_t, mask, points_3d = cv2.recoverPose(E, self.px_cur, self.px_ref, cameraMatrix = self.camera_matrix, distanceThresh=100)
        mask_bool = np.array(mask>0).reshape(-1)
        mask_e_bool = np.array(mask_e>0).reshape(-1)
        mask_bool = mask_bool & mask_e_bool
        points_3d_selected = points_3d[:,mask_bool].T
        points_3d_selected[:,0] = points_3d_selected[:,0]/points_3d_selected[:,3]
        points_3d_selected[:,1] = points_3d_selected[:,1]/points_3d_selected[:,3]
        points_3d_selected[:,2] = points_3d_selected[:,2]/points_3d_selected[:,3]

        self.feature3d = points_3d_selected[:,0:3]
        self.px_cur_selected = self.px_cur[mask_bool,:]
        self.px_ref_selected = self.px_ref[mask_bool,:]

        if(self.px_ref.shape[0] < kMinNumFeature):
            self.px_cur = self.detector.detect(self.new_frame)
            self.px_cur = np.array([x.pt for x in self.px_cur], dtype=np.float32)
            
        self.px_ref = self.px_cur

    def update(self, img, frame_id):
        assert(img.ndim ==2 and img.shape[0] == self.cam.height and img.shape[1] == self.cam.width), "Frame: provided image has not the same size as the camera model or image is not grayscale"
        self.new_frame = img
        if(self.frame_stage == STAGE_DEFAULT_FRAME):
            self.processFrame(frame_id)
        elif(self.frame_stage == STAGE_SECOND_FRAME):
            self.processSecondFrame()
        elif(self.frame_stage == STAGE_FIRST_FRAME):
            self.processFirstFrame()
        self.last_frame = self.new_frame

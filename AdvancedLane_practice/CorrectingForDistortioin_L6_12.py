import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt

#read in the saved objpoint and imgpoints

dist_pickle = pickle.load(open("wide_dist_pickle.p","rb"))
objpoints = dist_pickle["objpoints"]
imgpoints = dist_pickle["imgpoints"]

# read in an image
img = cv2.imread('test_image.png')

def cal_undistort(img,objpoints,imgpoints):
    # img.shape, it returns a tuple of number of rows, columns and channels
    img_size = (img.shape[1],img.shape[0])

    #cv2.calibrateCamera(), it retunrs the camera matrix, distortion, coefficients, rotation, and translation vectors
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    undist = cv2.undistort(img,mtx,dist,None,mtx)

    return undist

undistorted = cal_undistort(img,objpoints,imgpoints)

f,(ax1,ax2) = plt.subplots(1,2,figsize=(24,9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original_Image',fontsize = 50)
ax2.imshow(undistorted)
plt.show()
#adjust(left=0.,right=1,top=0.9,bottom=0.)

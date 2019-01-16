import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#read in the saved objpoint and imgpoints

dist_pickle = pickle.load(open("wide_dist_pickle.p","rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# read in an image
img = cv2.imread('test_image2.png')
nx = 8
ny = 6

def corners_unwarp(img,nx,ny,mtx,dist):
    # img.shape, it returns a tuple of number of rows, columns and channels
    undst  = cv2.undistort(img,mtx,dist,None,mtx)

    #cv2.calibrateCamera(), it retunrs the camera matrix, distortion, coefficients, rotation, and translation vectors
    gray = cv2.cvtColor(undst,cv2.COLOR_BGR2GRAY)
    ret,corners = cv2.findChessboardCorners(gray,(nx,ny),None)

    if ret == True:
        cv2.drawChessboardCorners(img,(nx,ny),corners,ret)
        plt.imshow(img)
        offset = 100
        img_size = (gray.shape[1],gray.shape[0])
        src = np.float32([corners[0],corners[nx-1],corners[-1],corners[-nx]])
        dst = np.float32([[offset,offset],[img_size[0]-offset,offset],[img_size[0]-offset,img_size[1]-offset],[offset,img_size[1]-offset]])

        M = cv2.getPerspectiveTransform(src,dst)
        warped = cv2.warpPerspective(undst,M,img_size,flags=cv2.INTER_LINEAR)
    return warped,M

top_down,perspective_M = corners_unwarp(img,nx,ny,mtx,dist)
f,(ax1,ax2) = plt.subplots(1,2,figsize = (24,9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original_Image',fontsize = 50)
ax2.imshow(top_down)
ax2.set_title('Undistorted and Warped Image',fontsize =50)
plt.show()

undistorted = cal_undistort(img,objpoints,imgpoints)

f,(ax1,ax2) = plt.subplots(1,2,figsize=(24,9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original_Image',fontsize = 50)
ax2.imshow(undistorted)
plt.show()
#adjust(left=0.,right=1,top=0.9,bottom=0.)

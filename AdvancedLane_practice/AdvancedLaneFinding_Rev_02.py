# Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
# Apply a distortion correction to raw images.
# Use color transforms, gradients, etc., to create a thresholded binary image.
# Apply a perspective transform to rectify binary image ("birds-eye view").
# Detect lane pixels and fit to find the lane boundary.
# Determine the curvature of the lane and vehicle position with respect to center.
# Warp the detected lane boundaries back onto the original image.
# Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

import numpy as np
import cv2
import pickle
import glob
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg

dist_pickle = pickle.load(open('calibration_pickle.p','rb'))


mtx = dist_pickle['mtx']
dist = dist_pickle['dist']


def abs_sobel_thresh(img,orient='x',sobel_kernel=3,thresh=(0,255)):
    # calculate directional gradient
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    if orient =='x':
        abs_sobel = np.absolute(cv2.Sobel(gray,cv2.CV_64F,1,0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray,cv2.CV_64F,0,1))

    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)

    binary_output[(scaled_sobel>=thresh[0])&(scaled_sobel<=thresh[1])] = 1
    return binary_output


def mag_thresh(image,sobel_kernel = 3,mag_thresh =(0,255)):
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize = sobel_kernel)
    sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize = sobel_kernel)
    gradmag = np.sqrt(sobelx**2,sobely**2)
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.unit8)
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return binary_output

def dir_threshold(image, sobel_kernel = 3, thresh = (0,np.pi/2)):
    # calculate gradient direction
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize = sobel_kernel)
    sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize = sobel_kernel)
    with np.errstate(divide='ignore',invaid='ignore'):
        absgraddir = np.absolute(np.arctan(sobely/sobelx))
        binary_output = np.zeros_like(absgraddir)
        #apply threshold
        binary_output[(absgraddir >= thresh[0])&(absgraddir <= thresh[1])]=1
    return binary_output

def color_threshold(image,sthresh = (0,255),vthresh = (0,255)):
    hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= sthresh[0])&(s_channel <= sthresh[1])] = 1

    hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    v_channel = hsv[:,:,2]
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel >=vthresh[0])&(v_channel <= vthresh[1])] = 1

    output = np.zeros_like(s_channel)
    output[(s_binary ==1)&(v_binary ==1)] = 1
    return output

def window_mask(width,height,img_ref,center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center_width/2),img_ref.shape[1])] = 1
    return output

# make a list of test images
images = glob.glob('test_images/test*.jpg')


for idx,fname in enumerate(images):
    # read in image
    img = cv2.imread(fname)
    # undistort the image
    img = cv2.undistort(img,mtx,dist,None,mtx)


    preprocessImage = np.zeros_like(img[:,:,0])
    gradx = abs_sobel_thresh(img, orient = 'x', thresh =(12,255))
    grady = abs_sobel_thresh(img, orient = 'y', thresh =(25,255))
    c_binary = color_threshold(img, sthresh = (100,255), vthresh = (50,255))
    preprocessImage[((gradx == 1)&(grady == 1)|(c_binary == 1))] = 255

    img_size =(img.shape[1],img.shape[0])
    bot_width = .76
    mid_width = .08
    height_pct= .62
    bottom_trim= .935


    offset = img_size[0]*.25
    src = np.float32([[img.shape[1]*(.5-mid_width/2),img.shape[0]*height_pct],[img.shape[1]*(.5*mid_width/2),img.shape[0]*height_pct],[img.shape[1]*(.5+bot_width/2),img.shape[0]*bottom_trim],[img.shape[1]*(.5+bot_width/2),img.shape[0]*bottom_trim]])
    dst = np.float32([[offset, 0], [img_size[0]-offset, offset],[img_size[0]-offset, img_size[1]-offset],[offset, img_size[1]-offset]])

    #M = cv2.getPerspectiveTransform(src, dst)
    #Minv = cv2.getPerspectiveTransform(dst, src)
    #warped = cv2.warpPerspective(undst, M, img_size, flags=cv2.INTER_LINEAR)


    result = preprocessImage
    write_name = 'test_images/tracked'+str(idx)+'.jpg'

    cv2.imwrite(write_name,result)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import cv2

# read in a thresholded image
warped = mpimg.imread('warped_example.jpg')
# window settings
window_width = 50
window_height = 80 # break image into 9 vertical layers since image height is 720
margin = 100 # how much to slide left and right for searching

def window_mask(width,height,img_ref,center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center_width/2),img_ref.shape[1])] = 1
    return output


def find_window_centroids(image,window_width,window_height,margin):
    window_centroids =[]
    window = np.ones(window_width)
    

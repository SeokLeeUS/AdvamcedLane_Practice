# Lesson 3 computer vision fundamentals
# color selection example

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Read in the image and print out some stats

image = mpimg.imread('test.jpg')
print('this image is:',type(image),'with dimensions:',image.shape)

# grab the x and y size and make a copy of the image
ysize = image.shape[0]
xsize = image.shape[1]

# define color selection criteria

red_threshold = 0
green_threshold = 0
blue_threshold = 0
rgb_threshold = [red_threshold,green_threshold,blue_threshold]

# note: always make a copy rather than simply using "="
color_select = np.copy(image)
line_image = np.copy(image)
region_select = np.copy(image)
# define color selection criteria

left_bottom = [0,539]
right_bottom = [900,300]
apex = [400,0]
# Fit lines(to identify 3 sided region of interest)
# np.polyfit() returns the coefficients [A,B] of the Fit

fit_left = np.polyfit((left_bottom[0],apex[0]),(left_bottom[1],apex[1]),1)
fit_right = np.polyfit((right_bottom[0],apex[0]),(right_bottom[1],apex[1]),1)
fit_bottom = np.polyfit((left_bottom[0],right_bottom[0]),(left_bottom[1],right_bottom[1]),1)


# Identify pixels below the threshold
color_thresholds = (image[:,:,0] < rgb_threshold[0]) \
            | (image[:,:,1] < rgb_threshold[1]) \
            | (image[:,:,2] < rgb_threshold[2])
color_select[color_thresholds] = [0,0,0]

#find the region inside the lines
XX,YY = np.meshgrid(np.arange(0,xsize),np.arange(0,ysize))
region_thresholds = (YY >(XX*fit_left[0])+fit_left[1])&\
                    (YY > (XX*fit_right[0] + fit_right[1])) & \
                    (YY < (XX*fit_bottom[0] + fit_bottom[1]))

#region_select[region_thresholds] = [255,0,0]

line_image[~color_thresholds&region_thresholds]=[255,0,0]

#display the image
plt.imshow(color_select)
plt.imshow(line_image)
plt.show()

#display the image
#plt.imshow(color_select)
#plt.show()

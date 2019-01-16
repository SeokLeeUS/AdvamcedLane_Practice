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

# note: always make a copy rather than simply using "="
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

#find the region inside the lines
XX,YY = np.meshgrid(np.arange(0,xsize),np.arange(0,ysize))
region_thresholds = (YY >(XX*fit_left[0])+fit_left[1])&\
                    (YY > (XX*fit_right[0] + fit_right[1])) & \
                    (YY < (XX*fit_bottom[0] + fit_bottom[1]))

region_select[region_thresholds] = [255,0,0]

#display the image
plt.imshow(region_select)
plt.show()

#display the image
#plt.imshow(color_select)
#plt.show()

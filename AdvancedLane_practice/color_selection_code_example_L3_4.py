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

color_select = np.copy(image)

# define color selection criteria

red_threshold = 0
green_threshold = 0
blue_threshold = 0
rgb_threshold = [red_threshold,green_threshold,blue_threshold]

# Identify pixels below the threshold
thresholds = (image[:,:,0] < rgb_threshold[0]) \
            | (image[:,:,1] < rgb_threshold[1]) \
            | (image[:,:,2] < rgb_threshold[2])
color_select[thresholds] = [0,0,0]

#display the image
plt.imshow(color_select)
plt.show()

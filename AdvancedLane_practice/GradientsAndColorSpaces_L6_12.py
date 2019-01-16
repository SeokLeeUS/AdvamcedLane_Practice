import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('bridge_shadow.jpg')

def pipeline(img,s_thresh=(170,255),sx_thresh=(20,100)):
    img = np.copy(img)
    hls = cv2.cvtColor(img,cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    sobelx = cv2.Sobel(l_channel,cv2.CV_64F,1,0)
    sobely = cv2.Sobel(s_channel,cv2.CV_64F,0,1)

    abs_sobex = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    abs_sobelxy = (sobelx**2+sobely**2)**0.5

    scaled_sobel = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))

    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel>=sx_thresh[0])&(s_channel<=s_thresh[1])] = 1

    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel>s_thresh[0])&(s_channel<=s_thresh[1])] = 1

    color_binary = np.dstack((np.zeros_like(sxbinary),sxbinary,s_binary))*255
    return color_binary

result = pipeline(image)

#plot result
f,(ax1,ax2) = plt.subplots(1,2,figsize=(24,9))
f.tight_layout()

ax1.imshow(image)
ax1.set_title('Original Image',fontsize = 40)

ax2.imshow(result)
ax2.set_title('Pipleline Result',fontsize = 40)

plt.show()

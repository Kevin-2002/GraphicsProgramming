# import OpenCV, numpy & matplotlib
import cv2
import numpy as np
from matplotlib import pyplot as plt

# read in image
imgOrig = cv2.imread('./ATU.jpg')

# read in a grayscale version
imgGray = cv2.cvtColor(imgOrig, cv2.COLOR_RGB2GRAY)

# use subplot to put multiple images in a single window 
plt.subplot(nrows, ncols,1),plt.imshow(imgOrig, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,2),plt.imshow(imgGray, cmap = 'gray')
plt.title(‘GrayScale’), plt.xticks([]), plt.yticks([])
plt.show()

# display image
cv2.imshow('Grayscale Image', imgGray)
cv2.waitKey(0) 
cv2.destroyAllWindows()

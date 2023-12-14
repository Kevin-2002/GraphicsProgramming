# import OpenCV, numpy & matplotlib
import cv2
import numpy as np
from matplotlib import pyplot as plt

# read in images
imgOrig = cv2.imread('./ATU.jpg')
myImg = cv2.imread('./myImg.jpeg')

# read in a grayscale version
imgGray = cv2.cvtColor(imgOrig, cv2.COLOR_RGB2GRAY)

# variables for subplotting
nrows = 3
ncols = 3

# blur variables
KernelSizeWidth = 3
KernelSizeHeight = 3

# store 3x3 blurred image 
imgBlur3x3 = cv2.GaussianBlur(imgGray,(KernelSizeWidth, KernelSizeHeight),0)

# change the blur variables to apply a 13x13 blur
KernelSizeWidth = 13
KernelSizeHeight = 13

# store 13x13 blurred image
imgBlur13x13 = cv2.GaussianBlur(imgGray,(KernelSizeWidth, KernelSizeHeight),0)

# sobel edge detection code 
imgSobelx = cv2.Sobel(imgGray,cv2.CV_64F,1,0,ksize=5) # x dir 
imgSobely = cv2.Sobel(imgGray,cv2.CV_64F,0,1,ksize=5) # y dir
imgSobelSum = imgSobelx + imgSobely # x&y dir

# canny variables
cannyThreshold = 100
cannyParam2 = 200

# canny edge detection code
imgCanny = cv2.Canny(imgGray,cannyThreshold,cannyParam2)

# use subplot to put multiple images in a single window 
plt.subplot(nrows, ncols,1),plt.imshow(cv2.cvtColor(imgOrig,
cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,2),plt.imshow(imgGray, cmap = 'gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,3),plt.imshow(imgBlur3x3, cmap = 'gray')
plt.title('3x3 Blur'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,4),plt.imshow(imgBlur13x13, cmap = 'gray')
plt.title('13x13 Blur'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,5),plt.imshow(imgSobelx, cmap = 'gray') 
plt.title('sobel x'), plt.xticks([]), plt.yticks([]) 
plt.subplot(nrows, ncols,6),plt.imshow(imgSobely, cmap = 'gray')
plt.title('sobel y'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,7),plt.imshow(imgSobelSum, cmap = 'gray')
plt.title('sobel Sum'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,8),plt.imshow(imgCanny, cmap = 'gray')
plt.title('Canny Trace'), plt.xticks([]), plt.yticks([])

plt.show()

# trial with myImg
# read in a grayscale version
myImgGray = cv2.cvtColor(myImg, cv2.COLOR_RGB2GRAY)

# store 3x3 blurred image 
myImgBlur3x3 = cv2.GaussianBlur(myImgGray,(KernelSizeWidth, KernelSizeHeight),0)

# store 13x13 blurred image
myImgBlur13x13 = cv2.GaussianBlur(myImgGray,(KernelSizeWidth, KernelSizeHeight),0)

# sobel edge detection code 
myImgSobelx = cv2.Sobel(myImgGray,cv2.CV_64F,1,0,ksize=5) # x dir 
myImgSobely = cv2.Sobel(myImgGray,cv2.CV_64F,0,1,ksize=5) # y dir
myImgSobelSum = myImgSobelx + myImgSobely # x&y dir

# canny edge detection code
MyImgCanny = cv2.Canny(myImgGray,cannyThreshold,cannyParam2)

# use subplot to put multiple images in a single window 
plt.subplot(nrows, ncols,1),plt.imshow(cv2.cvtColor(myImg,
cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,2),plt.imshow(myImgGray, cmap = 'gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,3),plt.imshow(myImgBlur3x3, cmap = 'gray')
plt.title('3x3 Blur'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,4),plt.imshow(myImgBlur13x13, cmap = 'gray')
plt.title('13x13 Blur'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,5),plt.imshow(myImgSobelx, cmap = 'gray') 
plt.title('sobel x'), plt.xticks([]), plt.yticks([]) 
plt.subplot(nrows, ncols,6),plt.imshow(myImgSobely, cmap = 'gray')
plt.title('sobel y'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,7),plt.imshow(myImgSobelSum, cmap = 'gray')
plt.title('sobel Sum'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,8),plt.imshow(MyImgCanny, cmap = 'gray')
plt.title('Canny Trace'), plt.xticks([]), plt.yticks([])

plt.show()


# display image
#cv2.imshow('Grayscale Image', imgGray)
#cv2.waitKey(0) 
#cv2.destroyAllWindows()

# import OpenCV, numpy & matplotlib
import cv2
import numpy as np
from matplotlib import pyplot as plt
import copy

# read in images
ATU1 = cv2.imread('./ATU1.jpg')
ATU2 = cv2.imread('./ATU2.jpg')
myImg = cv2.imread('./myImg.jpeg')

# make changes to images 
# read in a grayscale version
ATUGray1 = cv2.cvtColor(ATU1, cv2.COLOR_RGB2GRAY)
ATUGray2 = cv2.cvtColor(ATU2, cv2.COLOR_RGB2GRAY)
myImgGray = cv2.cvtColor(myImg, cv2.COLOR_RGB2GRAY)

# Harris corner detect
blockSize = 2
aperture_size = 3
k = 0.04

dst = cv2.cornerHarris(ATUGray1, blockSize, aperture_size, k)
#myimg
dst2 = cv2.cornerHarris(myImgGray, blockSize, aperture_size, k)

#deep copy
imgHarris = copy.deepcopy(ATUGray1)
#myimg
imgHarris2 = copy.deepcopy(myImgGray)

#set var
threshold = 0.5; #number between 0 and 1
R = 255
B = 50
G = 100
for i in range(len(dst)):
    for j in range(len(dst[i])):
        if dst[i][j] > (threshold*dst.max()):
            cv2.circle(imgHarris,(j,i),3,(B, G, R),-1)

#myimg
for i in range(len(dst)):
    for j in range(len(dst[i])):
        if dst[i][j] > (threshold*dst.max()):
            cv2.circle(imgHarris2,(j,i),3,(B, G, R),-1)

#shi tomasi
# set var
maxCorners = 200
qualityLevel = 0.01
minDistance = 10
corners = cv2.goodFeaturesToTrack(ATUGray1,maxCorners,qualityLevel,minDistance)
#myimg
corners = cv2.goodFeaturesToTrack(myImgGray,maxCorners,qualityLevel,minDistance)

# deep copy
imgShiTomasi = copy.deepcopy(ATUGray1)
#myimg
imgShiTomasi2 = copy.deepcopy(myImgGray)

# plot corners
R = 100
B = 100
G = 55
for i in corners:
    x,y = i.ravel()
    cv2.circle(imgShiTomasi,(int(x),int(y)),3,(B, G, R),-1)

#myimg
for i in corners:
    x,y = i.ravel()
    cv2.circle(imgShiTomasi2,(int(x),int(y)),3,(B, G, R),-1)

# orb
# Initiate ORB detector
orb = cv2.ORB_create()

# find the keypoints with ORB
kp = orb.detect(ATUGray1, None)
# myimg
kp2 = orb.detect(myImgGray, None)

# compute the descriptors with ORB
kp, des = orb.compute(ATUGray1, kp)
#myimg
kp2, des2 = orb.compute(myImgGray, kp2)

# draw keypoints location
imgOrb = cv2.drawKeypoints(ATUGray1, kp, None, color=(0, 255, 0), flags=0)
#myimg
imgOrb2 = cv2.drawKeypoints(myImgGray, kp2, None, color=(0, 255, 0), flags=0)

# variables for subplotting
nrows = 3
ncols = 3

# plot images
plt.subplot(nrows, ncols,1),plt.imshow(cv2.cvtColor(ATU1,
cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,2),plt.imshow(ATUGray1, cmap = 'gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,3),plt.imshow(dst, cmap = 'gray')
plt.title('Harris'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,4),plt.imshow(imgShiTomasi, cmap = 'gray')
plt.title('ShiTomasi'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,5),plt.imshow(imgOrb, cmap = 'gray')
plt.title('Orb'), plt.xticks([]), plt.yticks([])

# show images
plt.show()

# plot myimg
plt.subplot(nrows, ncols,1),plt.imshow(cv2.cvtColor(myImg,
cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,2),plt.imshow(myImgGray, cmap = 'gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,3),plt.imshow(dst2, cmap = 'gray')
plt.title('Harris'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,4),plt.imshow(imgShiTomasi2, cmap = 'gray')
plt.title('ShiTomasi'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,5),plt.imshow(imgOrb2, cmap = 'gray')
plt.title('Orb'), plt.xticks([]), plt.yticks([])

# show images
plt.show()

# advanced 1.
# canny edge detection code
cannyThreshold = 100
cannyParam2 = 200
MyImgCanny = cv2.Canny(myImgGray,cannyThreshold,cannyParam2)

# Find contours
contours, _ = cv2.findContours(MyImgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on a blank canvas
contour_img = np.zeros_like(myImg)

# Draw contours on the blank canvas
cv2.drawContours(contour_img, contours, -1, (255, 255, 255), thickness=1)

# Display the original image and the contours
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(myImg, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(contour_img, cmap='gray')
plt.title('Contours')
plt.axis('off')

plt.tight_layout()
plt.show()

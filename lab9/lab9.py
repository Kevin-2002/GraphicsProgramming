# import OpenCV, numpy & matplotlib
import cv2
import numpy as np
from matplotlib import pyplot as plt

#advanced 1.
# Find contours
# contours, _ = cv2.findContours(MyImgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on a blank canvas
# contour_img = np.zeros_like(myImg)

# Draw contours on the blank canvas
# cv2.drawContours(contour_img, contours, -1, (255, 255, 255), thickness=1)

# Display the original image and the contours
# plt.subplot(1, 2, 1)
# plt.imshow(cv2.cvtColor(myImg, cv2.COLOR_BGR2RGB))
# plt.title('Original Image')
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.imshow(contour_img, cmap='gray')
# plt.title('Contours')
# plt.axis('off')

# plt.tight_layout()
# plt.show()

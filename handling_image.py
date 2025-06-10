#!/usr/bin/env python

import cv2
img = cv2.imread('image.jpg')

# Crop image based on the array indices
img_cropped = img[10:50, 25:55]
cv2.imwrite('image_cropped.jpg', img_cropped)

# Increase the brightness
brightness = 100
img_bright = img.copy()
img_bright[img_bright > 255 - brightness] = 255 - brightness
img_bright += brightness
cv2.imwrite('image_light.jpg', img_bright)

# Resize
img_30x30 = cv2.resize(img, (30, 30))
cv2.imwrite('image_30x30.jpg', img_30x30)

# Resize and ensure aspect ratio is preserved
height, width = img.shape[:2]
new_width = 130
aspect_ratio = width / height
new_height = int(new_width / aspect_ratio)
img_resized = cv2.resize(img, (new_width, new_height))
cv2.imwrite('image_resized.jpg', img_resized)

# Changing colourspaces
## Aggregate all the colors based on their luminance
img_mean = img.mean(axis=2).astype('uint8')
cv2.imwrite('image_mean.jpg', img_mean)

## OpenCV change BGR to gray
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('image_gray.jpg', img_gray)

## OpenCV change BGR to HSV
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imwrite('image_hsv.jpg', img_hsv)

# OpenCV blurring
img_blur = cv2.GaussianBlur(img, (9, 9), 0)
cv2.imwrite('image_blur.jpg', img_blur)

# OpenCV sharpening
import numpy as np
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
img_gray_sharp = cv2.filter2D(img_gray, -1, kernel)
cv2.imwrite('image_gray_sharp.jpg', img_gray_sharp)

## Same with color image
img_sharp = cv2.filter2D(img, -1, kernel)
cv2.imwrite('image_sharp.jpg', img_sharp)

# OpenCV enhancing image contrast
img_equalized = cv2.equalizeHist(img_gray)
cv2.imwrite('image_equalized.jpg', img_equalized)

import matplotlib.pyplot as plt
plt.figure()
plt.hist(img_gray.flatten(), bins=100)
plt.xlim(0, 255)
plt.xlabel('Pixel intensity')
plt.ylabel('Pixel frequency')
plt.savefig('img_gray_hist.jpg')

plt.figure()
plt.hist(img_equalized.flatten(), bins=100)
plt.xlim(0, 255)
plt.xlabel('Pixel intensity')
plt.ylabel('Pixel frequency')
plt.savefig('img_equalized_hist.jpg')

# Normalisation
img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
cv2.imwrite('image_norm.jpg', img_norm)

import matplotlib.pyplot as plt
fig = plt.figure()
colours = ['blue', 'green', 'red']
for i in range(3):
    ax = fig.add_subplot(3, 1, i+1)
    ax.hist(img[:,:,i].flatten(), bins=100, color=colours[i])
    ax.set_xlim(0, 255)
    if i == 1:
        ax.set_ylabel('Pixel frequency')
    if i == 2:
        ax.set_xlabel('Pixel intensity')
plt.savefig('img_hist.jpg')

import matplotlib.pyplot as plt
fig = plt.figure()
colours = ['blue', 'green', 'red']
for i in range(3):
    ax = fig.add_subplot(3, 1, i+1)
    ax.hist(img_norm[:,:,i].flatten(), bins=100, color=colours[i])
    ax.set_xlim(0, 255)
    if i == 1:
        ax.set_ylabel('Pixel frequency')
    if i == 2:
        ax.set_xlabel('Pixel intensity')
plt.savefig('img_norm_hist.jpg')


# FIXME The values are centered at zero with negative and positive values
img_zscore = ((img - np.mean(img)) / np.std(img)).astype('uint8')
cv2.imwrite('image_zscore.jpg', img_zscore)

# OpenCV edge detection
img_edges = cv2.Canny(img_gray, 100, 200)
cv2.imwrite('image_edges.jpg', img_edges)

# OpenCV thresholding
ret, img_bin = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
cv2.imwrite('image_bin.jpg', img_bin)

# OpenCV adaptive threshold
img_adaptive = cv2.adaptiveThreshold(img_gray, maxValue=255, \
                                     adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                     thresholdType=cv2.THRESH_BINARY, \
                                     blockSize=5, C=5)
cv2.imwrite('image_adaptive.jpg', img_adaptive)

# OpenCV dilation
kernel = np.ones((3, 3), np.uint8)
img_dilated = cv2.dilate(img_bin, kernel, iterations=1)
cv2.imwrite('image_dilated.jpg', img_dilated)

# OpenCV erosion
kernel = np.ones((3, 3), np.uint8)
img_eroded = cv2.erode(img_bin, kernel, iterations=1)
cv2.imwrite('image_eroded.jpg', img_eroded)

# OpenCV feature detection
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(img, None)
img_keypoints = cv2.drawKeypoints(img, keypoints, None)
cv2.imwrite('image_keypoints.jpg', img_keypoints)

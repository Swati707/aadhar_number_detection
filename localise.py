# import the necessary packages
import numpy as np
import cv2
import imutils
import argparse
from perspectiveTransform import *
from skimage.filters import threshold_local

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the image to be localised")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

# if the width is greater than 640 pixels, then resize the image
if image.shape[1] > 640:
	image = imutils.resize(image, width=640)
	
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0)

edged = cv2.Canny(gray, 0, 50)
edgedi = cv2.bitwise_not(edged)
edgedi = cv2.GaussianBlur(edgedi, (5, 5), 0)
edgedi = cv2.erode(edgedi, None, iterations=3)
edgedit = cv2.threshold(edgedi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# show the original image and the edge detected image
#edgedit = cv2.bitwise_not(edgedit)
print("STEP 1: Edge Detection")
#cv2.imshow("Edged", edgedit)

output = np.copy(image)
im2, contours, hierarchy = cv2.findContours(edgedit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print(len(contours))
if len(contours) != 0:
    # draw in blue the contours that were founded
    #cv2.drawContours(output, contours, -1, 255, 3)
    if len(contours) == 1:
    	im2,contours,hierarchy = cv2.findContours(cv2.bitwise_not(edgedit), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # find the biggest countour (c) by the area
    c = max(contours, key = cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)

    # draw the biggest contour (c) in green
    cv2.rectangle(output,(x,y),(x+w,y+h+3),(0,0,255),3)

cv2.imshow("Edged", output)
cv2.waitKey(0)

crop_img = image[y:y+h, x:x+w]
cv2.imshow("Crop", crop_img)
#cv2.waitKey(0)

image = crop_img
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 9))
squareKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
regions = []

# convert the image to grayscale, and apply the blackhat operation
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)

#cv2.imshow("gray", gray)
#cv2.imshow("blackhat", blackhat)

#cv2.waitKey(0)

# compute the Scharr gradient representation of the blackhat image in the x-direction,
# and scale the resulting image into the range [0, 255]
gradX = cv2.Sobel(blackhat,
	ddepth=cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F,
	dx=1, dy=0, ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

# blur the gradient representation, apply a closing operation, and threshold the
# image using Otsu's method
gradX = cv2.GaussianBlur(gradX, (3, 3), 0)
cv2.imshow("gradXgb", gradX)

gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

cv2.imshow("gradXClosed", gradX)
cv2.imshow("thresh", thresh)
cv2.waitKey(0)

# perform a series of erosions and dilations on the image
thresh = cv2.dilate(thresh, None, iterations=4)
thresh = cv2.erode(thresh, None, iterations=1)
cv2.imshow("dilate_erode", thresh)
cv2.waitKey(0)

thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, rectKernel)
cv2.imshow("threshOpen", thresh)
cv2.waitKey(0)

contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contours = imutils.grab_contours(contours)

def draw_contour(image, c, i):
    # compute the center of the contour area and draw a circle
    # representing the center
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    # draw the countour number on the image
    cv2.putText(image, "#{}".format(i + 1), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX,
        1.0, (255, 255, 255), 2)
    # return the image with the contour number drawn on it
    return image

def sort_contours(cnts, method="bottom-to-top"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
        key=lambda b:b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

(cnts, boundingBoxes) = sort_contours(contours)

orig = image.copy()
for (i, c) in enumerate(cnts):
    draw_contour(orig, c, i)

# show the output image
cv2.imshow("Sorted", orig)
cv2.waitKey(0)

regions = []
for c in cnts[:3]:
    # grab the bounding box associated with the contour and compute the area and
    # aspect ratio
    (w, h) = cv2.boundingRect(c)[2:]
    aspectRatio = w / float(h)

    # compute the rotated bounding box of the region
    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.cv.BoxPoints(rect)) if imutils.is_cv2() else cv2.boxPoints(rect)

    shapeArea = cv2.contourArea(c)
    bboxArea = w * h
    extent = shapeArea / float(bboxArea)
    extent = int(extent * 100) / 100
    print(aspectRatio, extent)
    # ensure the aspect ratio, width, and height of the bounding box fall within
    # tolerable limits, then update the list of license plate regions
    if (aspectRatio > 5 and aspectRatio < 10) and extent > 0.65:
        regions.append(box)

potential_detection = image.copy()

for lpBox in regions:
    lpBox = np.array(lpBox).reshape((-1,1,2)).astype(np.int32)
    cv2.drawContours(potential_detection, [lpBox], -1, (0, 255, 0), 2)

# display the output image
cv2.imshow("image", potential_detection)
cv2.waitKey(0)

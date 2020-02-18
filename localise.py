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
cv2.imshow("Edged", edgedit)
cv2.waitKey(0)
cv2.destroyAllWindows()
output = np.copy(image)
im2,contours,hierarchy = cv2.findContours(edgedit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
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
    cv2.rectangle(image,(x,y),(x+w,y+h+3),(0,0,255),3)

cv2.imshow("Edged", image)
cv2.waitKey(0)

rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
squareKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
regions = []

# convert the image to grayscale, and apply the blackhat operation
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)

cv2.imshow("gray", gray)
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

#cv2.imshow("gradX", gradX)
#cv2.waitKey(0)

# blur the gradient representation, apply a closing operation, and threshold the
# image using Otsu's method
gradX = cv2.GaussianBlur(gradX, (3, 3), 0)
#cv2.imshow("gradX", gradX)


gradX = cv2.erode(gradX, None, iterations=2)
#cv2.imshow("erode", gradX)

gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

#cv2.imshow("gradXMorph", gradX)
#cv2.imshow("thresh", thresh)
#cv2.waitKey(0)

# perform a series of erosions and dilations on the image
thresh = cv2.dilate(thresh, None, iterations=3)
thresh = cv2.erode(thresh, None, iterations=2)
cv2.imshow("dilate", thresh)
cv2.waitKey(0)

thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, rectKernel)
cv2.imshow("threshOpen", thresh)
cv2.waitKey(0)

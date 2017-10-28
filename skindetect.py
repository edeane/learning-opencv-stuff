#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 20:59:21 2017

@author: applesauce

https://www.pyimagesearch.com/2014/08/18/skin-detection-step-step-example-using-python-opencv/
https://www.pyimagesearch.com/2014/08/04/opencv-python-color-detection/

https://docs.opencv.org/trunk/dc/df6/tutorial_py_histogram_backprojection.html
"""

import numpy as np
import cv2

frame = None
roiPts = []
inputMode = False
lower = np.array([0, 0, 0], dtype = "uint8")
upper = np.array([0, 0, 0], dtype = "uint8")

def resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized

def selectROI(event, x, y, flags, param):
    global frame, roiPts, inputMode, lower, upper
    if inputMode and event == cv2.EVENT_LBUTTONDOWN and len(roiPts) < 4:
        roiPts.append((x, y))
        cv2.circle(frame, (x, y), 4, (0, 255, 0), 2)
        cv2.imshow("frame", frame)

camera = cv2.VideoCapture(0)
cv2.namedWindow("frame")
cv2.setMouseCallback("frame", selectROI)
roiBox = None

while True:
    (grabbed, frame) = camera.read()
    if not grabbed:
        break
#    frame = resize(frame, width = 500)
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    #skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    #skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
    #skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(frame, frame, mask = skinMask)
    if len(roiPts) < 4:
        cv2.imshow("frame", frame)
    else:
        cv2.imshow("frame", np.hstack([frame, converted, skin]))
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("i") and len(roiPts) < 4:
        inputMode = True
        orig = frame.copy()
        while len(roiPts) < 4:
            cv2.imshow("frame", frame)
            cv2.waitKey(0)
        roiPts = np.array(roiPts)
        s = roiPts.sum(axis = 1)
        tl = roiPts[np.argmin(s)]
        br = roiPts[np.argmax(s)]
        roi = orig[tl[1]:br[1], tl[0]:br[0]]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        #b g r
#        roi_2d = roi.sum(axis=2)
#        lower = roi[np.unravel_index(roi_2d.argmin(), roi_2d.shape)]
#        upper = roi[np.unravel_index(roi_2d.argmax(), roi_2d.shape)]
        lower_blue = roi[:,:,0].min()
        lower_green = roi[:,:,1].min()
        lower_red = roi[:,:,2].min()
        upper_blue = roi[:,:,0].max()
        upper_green = roi[:,:,1].max()
        upper_red = roi[:,:,2].max()
        lower = np.array((lower_blue, lower_green, lower_red))
        upper = np.array((upper_blue, upper_green, upper_red))
        print('lower {}'.format(lower))
        print('upper {}'.format(upper))
    elif key == ord("q"):
        break
camera.release()
cv2.destroyAllWindows()
    

    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 20:59:21 2017

@author: applesauce

https://www.pyimagesearch.com/2014/08/18/skin-detection-step-step-example-using-python-opencv/
https://www.pyimagesearch.com/2014/08/04/opencv-python-color-detection/
"""

import numpy as np
import cv2

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

def skin_detect():
    # B G R
    lower = np.array([80, 40, 60], dtype = "uint8")
    upper = np.array([220, 110, 160], dtype = "uint8")
    camera = cv2.VideoCapture(0)
    while True:
    	(grabbed, frame) = camera.read()
    	if not grabbed:
    		break
    	frame = resize(frame, width = 400)
    	converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    	skinMask = cv2.inRange(converted, lower, upper)
    	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    #	skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    #	skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
    	skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    	skin = cv2.bitwise_and(frame, frame, mask = skinMask)
     
    	# show the skin in the image along with the mask
    	cv2.imshow("images", np.hstack([frame, converted, skin]))
     
    	# if the 'q' key is pressed, stop the loop
    	if cv2.waitKey(1) & 0xFF == ord("q"):
    		break    
    camera.release()
    cv2.destroyAllWindows()
    


if __name__ == '__main__':
    skin_detect()
    
    
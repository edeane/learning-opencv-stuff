#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 18:06:27 2017

@author: applesauce
https://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/
"""


import numpy as np
import cv2
from collections import deque
import time

### add select ball color

def selectROI(event, x, y, flags, param):
    global frame, roiPts, inputMode, lower, upper
    if inputMode and event == cv2.EVENT_LBUTTONDOWN and len(roiPts) < 4:
        roiPts.append((x, y))
        cv2.circle(frame, (x, y), 4, (0, 255, 0), 2)
        cv2.imshow("frame", frame)

inputMode = False
lower = np.array([0, 0, 0], dtype = "uint8")
upper = np.array([0, 0, 0], dtype = "uint8")
buffer = 30
pts = deque(maxlen=buffer)
cv2.namedWindow('frame')
cv2.setMouseCallback('frame', selectROI)
roiBox = None
frames = 0
fps = 0
frame = None
roiPts = []
camera = cv2.VideoCapture(0)

        
while True:
    if frames == 0:
        start = time.time()
    frames+=1
    grabbed, frame = camera.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None
    if len(cnts)>0:
        c = max(cnts, key=cv2.contourArea)
        ((x,y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0,255,255), 2)
            cv2.circle(frame, center, 5, (0,0,255), -1)
        pts.appendleft(center)
        for i in range(1, len(pts)):
            if pts[i-1] is None or pts[i] is None:
                continue
            thickness = int(np.sqrt(buffer/(i+1)*2.5))
            cv2.line(frame, pts[i-1], pts[i], (0,0,255), thickness)
    end = time.time()
    time_dif = end - start
    if time_dif >= 1:
        fps = round(frames / time_dif, 2)
        frames=0
    cv2.putText(frame, 'fps: {}'.format(str(fps)), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    cv2.imshow('frame', np.hstack([frame, hsv, mask_bgr]))
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('i'):
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


    
camera.release()
cv2.destroyAllWindows()


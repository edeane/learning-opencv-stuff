#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 21:48:45 2017

@author: applesauce

https://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/


todo:
    
fix box somehow


"""
import numpy as np
import cv2

image = cv2.imread('images/textskew/pos_41.png')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray)
#gray = cv2.GaussianBlur(gray, (9,9), 0)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]



coords = np.column_stack(np.where(thresh > 0))
angle = cv2.minAreaRect(coords)[-1]

x_max = coords[:,0].max()
x_min = coords[:,0].min()
y_max = coords[:, 1].max()
y_min = coords[:, 1].min()

xs = (x_max, x_min)
ys = (y_max, y_min)

coords_sum = coords.sum(axis=1)

cv2.circle(image, tuple(coords[coords[:,0].argmin(),:]), 4, (0, 255, 0), 2)
cv2.circle(image, tuple(coords[coords[:,1].argmin(),:]), 4, (0, 255, 0), 2)
cv2.circle(image, tuple(coords[coords[:,0].argmax(),:]), 4, (0, 255, 0), 2)
cv2.circle(image, tuple(coords[coords[:,1].argmax(),:]), 4, (0, 255, 0), 2)

rect = cv2.minAreaRect(coords)

box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(image,[box],-1,(0,255,0),2)

rows,cols = thresh.shape[:2]
[vx,vy,x,y] = cv2.fitLine(coords, cv2.DIST_L2,0,0.01,0.01)
lefty = int((-x*vy/vx) + y)
righty = int(((cols-x)*vy/vx)+y)
cv2.line(image,(cols-1,righty),(0,lefty),(0,255,0),2)

#(_, cnts, _) = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:1]
#
#for cnt in cnts:
#    rect = cv2.minAreaRect(cnt)
#    box = cv2.boxPoints(rect)
#    box = np.int0(box)
#    cv2.drawContours(image,[box],0,(0,0,255),2)


print(angle)
if angle < -45:
	angle = -(90 + angle)
else:
	angle = -angle
print(angle)
(h, w) = image.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#print("[INFO] angle: {:.3f}".format(angle))
cv2.imshow("Input", image)
cv2.imshow("Rotated", rotated)

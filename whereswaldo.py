#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 14:17:46 2017

@author: applesauce

https://www.pyimagesearch.com/wp-content/uploads/2014/11/opencv_crash_course_waldo.pdf?__s=ugii8jwnb5wda2ppoczc
"""

import numpy as np
import cv2

def waldo():
    find_waldo = cv2.imread('images/waldo/puzzle_small.jpg')
    cv2.imshow('find waldo', find_waldo)
    tl = (98,569)
    br = (111,597)
    waldo_query = find_waldo[tl[1]:br[1], tl[0]:br[0]]
    cv2.imshow('waldo query', waldo_query)
    (waldoHeight, waldoWidth) = waldo_query.shape[:2]
    result = cv2.matchTemplate(find_waldo, waldo_query, cv2.TM_CCOEFF)
    (_, _, minLoc, maxLoc) = cv2.minMaxLoc(result)
    topLeft = maxLoc
    botRight = (topLeft[0] + waldoWidth, topLeft[1] + waldoHeight)
    roi = find_waldo[topLeft[1]:botRight[1], topLeft[0]:botRight[0]]
    mask = np.zeros(find_waldo.shape, dtype = "uint8")
    find_waldo = cv2.addWeighted(find_waldo, 0.25, mask, 0.75, 0)
    find_waldo[topLeft[1]:botRight[1], topLeft[0]:botRight[0]] = roi
    cv2.imshow('solution', find_waldo)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    waldo()
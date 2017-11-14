#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 17:21:43 2017

@author: applesauce

https://www.pyimagesearch.com/2016/01/11/opencv-panorama-stitching/
"""

import cv2
import numpy as np

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

class Stitcher():
    
    def stitch(self, images, ratio=.75, reprojThresh=4.0, showMatches=False):
        imageB, imageA = images
        kpsA, featuresA = self.detectAndDescribe(imageA)
        kpsB, featuresB = self.detectAndDescribe(imageB)
        
        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)
        
        if M is None:
            return None
        
        matches, H, status = M
        result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
        
        if showMatches:
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
            return result, vis
        
        return result
    
    def detectAndDescribe(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        descriptor = cv2.xfeatures2d.SIFT_create()
        kps, features = descriptor.detectAndCompute(gray, None)
        kps = np.float32([kp.pt for kp in kps])
        return kps, features
    
    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
        matcher = cv2.DescriptorMatcher_create('BruteForce')
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []
        for m in rawMatches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))
        if len(matches) > 4:
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])
                
            H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
            return matches, H, status
        return None
        
    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        hA, wA = imageA.shape[:2]
        hB, wB = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA+wB, 3), dtype='uint8')
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            if s == 1:
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
                
        return vis
    
imageA = cv2.imread('images/panorama/grand_canyon_left_01.png')
imageB = cv2.imread('images/panorama/grand_canyon_right_01.png')
imageA = resize(imageA, width=400)
imageB = resize(imageB, width=400)

stitcher = Stitcher()
result, vis = stitcher.stitch([imageA, imageB], showMatches=True)

cv2.imshow('image a', imageA)
cv2.imshow('image b', imageB)
cv2.imshow('keypoint matches', vis)
cv2.imshow('result', result)





        
        
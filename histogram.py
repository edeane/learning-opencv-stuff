#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 10:22:53 2017

@author: applesauce

https://www.pyimagesearch.com/2014/01/22/clever-girl-a-guide-to-utilizing-color-histograms-for-computer-vision-and-image-search-engines/
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def gray_hist(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', gray)
    hist_1 = cv2.calcHist([gray], [0], None, [256], [0, 256])
    fig, ax = plt.subplots(1, 1)
    ax.plot(hist_1)
    ax.set_xlabel('bins')
    ax.set_ylabel('# of pixels')
    ax.set_xlim([0, 256])
    plt.title('grayscale histogram 1')
    plt.show()
    hist_2 = np.histogram(gray, bins=255)
    fig, ax = plt.subplots(1, 1)
    ax.plot(hist_2[0])
    ax.set_xlabel('bins')
    ax.set_ylabel('# of pixels')
    ax.set_xlim([0, 256])
    plt.title('grayscale histogram 2')
    plt.show()


def flat_hist(image):
    chans = cv2.split(image)
    colors = ('b', 'g', 'r')
    plt.figure()
    plt.title('flattened color hist')
    plt.xlabel('bins')
    plt.ylabel('# of pixels')
    features = []
    for (chan, color) in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], None, [256], [0,256])
        features.append(hist)
        plt.plot(hist, color=color)
        plt.xlim([0,256])
    print('flattened feature vector size: {}'.format(np.array(features).flatten().shape))

def two_d_hist(image):
    chans = cv2.split(image)
    combos = [(0, 1), (0, 2), (1, 2)]
    combos_names = ['blue', 'green', 'red']
    fig, axes = plt.subplots(1, 3)
    for (a, b), ax in zip(combos, axes.reshape(-1)):
        hist = cv2.calcHist([chans[a], chans[b]], [0, 1], None, [32, 32], [0, 256, 0, 256])
        ax.imshow(hist, interpolation='nearest')
        ax.set_title('{} and {}'.format(combos_names[a], combos_names[b]))
    plt.show()



image = cv2.imread('images/grant.jpg')
chans = cv2.split(image)      


hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
plt.plot(hist)
plt.show()

           
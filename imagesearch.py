#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 17:54:00 2017

@author: applesauce

https://www.pyimagesearch.com/2014/01/27/hobbits-and-histograms-a-how-to-guide-to-building-your-first-image-search-engine-in-python/
"""

import numpy as np
import cv2
import os
import pickle


class RGBHistogram():
    def __init__(self, bins):
        self.bins = bins
        
    def describe(self, image):
        hist = cv2.calcHist([image], [0, 1, 2], None, self.bins, [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist)
        return hist.flatten()
    

def create_index():
    index = {}
    desc = RGBHistogram([8, 8, 8])
    dataset_path = 'images/imagesearch/images/'
    dataset_list = [f for f in os.listdir(dataset_path) if f.endswith('.png')]
    for file_name in dataset_list:
        image = cv2.imread(dataset_path+file_name)
        features = desc.describe(image)
        index[file_name] = features        
    pickle.dump(index, open('images/imagesearch/index.pkl', 'wb'))


def import_pickle():
    index = pickle.load(open('images/imagesearch/index.pkl', 'rb'))
    print(index.keys())
    
    

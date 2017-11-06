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

### DESCRIPTOR
class RGBHistogram():
    def __init__(self, bins):
        self.bins = bins
        
    def describe(self, image):
        hist = cv2.calcHist([image], [0, 1, 2], None, self.bins, [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist)
        return hist.flatten()
    
### INDEX
def create_index(images_or_queries='images'):
    index = {}
    desc = RGBHistogram([8, 8, 8])
    dataset_path = 'images/imagesearch/'+images_or_queries+'/'
    dataset_list = [f for f in os.listdir(dataset_path) if f.endswith('.png')]
    for file_name in dataset_list:
        image = cv2.imread(dataset_path+file_name)
        features = desc.describe(image)
        index[file_name] = features        
    return index

### SEARCH
class Searcher():
    def __init__(self, index):
        self.index = index
    
    def search(self, queryFeatures):
        results = {}
        for k, features in self.index.items():
            d = self.chi2_distance(features, queryFeatures)
            results[k] = d
        results = sorted([(v, k) for (k, v) in results.items()])
        return results
    def chi2_distance(self, histA, histB, eps=1e-10):
        d = .5*np.sum([((a-b)**2) / (a+b+eps) for (a, b) in zip(histA, histB)])
        return d
            

### SEARCHING
index = pickle.load(open('images/imagesearch/index.pkl', 'rb'))
searcher = Searcher(index)

images_or_queries = 'queries'
search_images = create_index(images_or_queries)
dataset_path = 'images/imagesearch/'+images_or_queries+'/'

for (query, queryFeatures) in search_images.items():
    results = searcher.search(queryFeatures)
    path = dataset_path + query
    queryImage = cv2.imread(path)
    cv2.imshow('query', queryImage)
    montageA = np.zeros((166*5, 400, 3), dtype='uint8')
    montageB = np.zeros((166*5, 400, 3), dtype='uint8')
    for j in range(0, 10):
        (score, imageName) = results[j]
        path = 'images/imagesearch/images/' + imageName
        result = cv2.imread(path)
        print('{}. {} : {}'.format(j+1, imageName, score))
        if j < 5:
            montageA[j*166:(j+1)*166, :] = result
        else:
            montageB[(j-5)*166:((j-5)+1)*166, :] = result
    cv2.imshow('results 1-5', montageA)
    cv2.imshow('results 6-10', montageB)
    cv2.waitKey(0)
cv2.destroyAllWindows()







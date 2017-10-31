#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 15:00:47 2017

@author: applesauce

https://www.pyimagesearch.com/2014/05/26/opencv-python-k-means-color-clustering/
"""

import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

image = cv2.imread('images/kmeans/thematrix.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure()
plt.axis("off")
plt.imshow(image)
plt.show()

n_clusters = 5

image = image.reshape((image.shape[0] * image.shape[1], 3))
clt = KMeans(n_clusters)
clt.fit(image)

clusters = clt.cluster_centers_

fig, axes = plt.subplots(n_clusters, 1)

for index, ax in enumerate(axes.reshape(-1)):
    square = np.array([clusters[index].astype('uint8')] * 10000).reshape(100,100,3)
    ax.imshow(square)
    ax.set_axis_off()
    
plt.show()
    
def centroid_histogram(clt):
	(hist, _) = np.histogram(clt.labels_, bins = clt.n_clusters)
	hist = hist.astype("float")
	hist /= hist.sum()
	return hist

def plot_colors(hist, centroids):
	bar = np.zeros((50, 300, 3), dtype = "uint8")
	startX = 0
	for (percent, color) in zip(hist, centroids):
		endX = startX + (percent * 300)
		cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
			color.astype("uint8").tolist(), -1)
		startX = endX
	return bar

hist = centroid_histogram(clt)
bar = plot_colors(hist, clt.cluster_centers_)
plt.figure()
plt.axis("off")
plt.imshow(bar)
plt.show()


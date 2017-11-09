#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 16:29:11 2017

@author: applesauce
"""

import numpy as np
import cv2
from bs4 import BeautifulSoup
import requests
import mahotas
import os
import pickle
from skimage import exposure
from scipy.spatial import distance


def scrape_poke():
    url = 'https://pokemondb.net/pokedex/national'
    response = requests.get(url)
    print(response.status_code)
    soup = BeautifulSoup(response.content, 'html.parser')
    names = []
    for link in soup.findAll('a'):
        names.append(link.text)        
    for name in names:
        parsedName = name.lower()
        parsedName = parsedName.replace("'","")
        parsedName = parsedName.replace('. ', '-')
        parsedName = parsedName.replace(' ', '-')
        if name.find(u'\u2640') != -1:
            parsedName = 'nidoran-f'
        elif name.find(u'\u2642') != -1:
            parsedName = 'nidoran-m'
        url = 'https://img.pokemondb.net/sprites/red-blue/normal/{}.png'.format(parsedName)
        r = requests.get(url)
        if r.status_code != 200:
            print('[x] error downloading {}'.format(parsedName))
            continue
        print('[x] downloading {}'.format(parsedName))
        f = open('images/pokedex/sprites/red-blue/{}.png'.format(name.lower()), 'wb')
        f.write(r.content)
        f.close()
        
def my_test():
    image = cv2.imread('images/pokedex/sprites/eevee.jpg')
    image = cv2.copyMakeBorder(image, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=(255,255,255))
    cv2.imshow('image', image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', gray)
    retval, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    cv2.imshow('thresh', thresh)
    (_, contours, _) = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:3]
    cnt = contours[1]
    M = cv2.moments(cnt)
    epsilon = 0.1*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
    cv2.imshow('cnt', image)

class ZernikeMoments():
    def __init__(self, radius):
        self.radius = radius
        
    def describe(self, image):
        return mahotas.features.zernike_moments(image, self.radius)
    

def create_index():
    desc = ZernikeMoments(21)
    index = {}
    dataset_path = 'images/pokedex/sprites/red-blue/'
    dataset_list = [f for f in os.listdir(dataset_path) if f.endswith('.png')]
    for i, poke in enumerate(dataset_list):
        image = cv2.imread(dataset_path+poke)
        outline = create_outline(image)
        moments = desc.describe(outline)
        index[poke] = moments
    pickle.dump(index, open('images/pokedex/index.pkl', 'wb'))
    
    
def create_outline(image):
    image = cv2.copyMakeBorder(image, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=(255,255,255))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #gray = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.bitwise_not(gray)
    thresh[thresh>10] = 255
    thresh[thresh<10] = 0
    outline = np.zeros(thresh.shape, dtype='uint8')
    (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    cv2.drawContours(outline, [cnts], -1, (255, 255, 255), -1)
    return outline
    
    
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


def get_gameboy():
    image = cv2.imread('images/pokedex/gameboy/query_kadabra.jpg')
    ratio = image.shape[0] / 300
    orig = image.copy()
    image = resize(image, height=300)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bifil = cv2.bilateralFilter(gray, 11, 17, 17) #similar to blur but keeps edges better
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edged_bifil = cv2.Canny(bifil, 20, 250)
    (_, cnts, _) = cv2.findContours(edged_bifil, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv2.contourArea, reverse=True)[:100]
    screenCnt = None
    for i, cnt in enumerate(cnts):
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
        print(i)
        if len(approx) == 4:
            screenCnt = approx
            break
    cv2.drawContours(image, [screenCnt], -1, (255,0,0), 2)
    pts = screenCnt.reshape(4,2)
    rect = np.zeros((4,2), dtype='float32')
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    rect *= ratio
    tl, tr, br, bl = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0,0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]], dtype='float32')
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))
    warp_before = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    warp_after = exposure.rescale_intensity(warp_before, out_range=(0,255))
    h, w = warp_after.shape
    dX, dY = (int(w*.4), int(h*.45))
    crop = warp_after[10:dY, w-dX:w-10]
    cv2.imshow('crop', crop)
    cv2.imwrite('images/pokedex/gameboy/query.png', crop)


def show_xy():
    xy = np.zeros((300, 500, 3))
    locations = ((10,20), (390,290), (10,290), (390,20))
    for loc in locations:
        cv2.putText(xy, str(loc), loc, cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 2)
    cv2.imshow('xy', xy)


class Searcher():
    def __init__(self, index):
        self.index = index
    def search(self, queryFeatures):
        results = {}
        for k, features in self.index.items():
            d = distance.euclidean(queryFeatures, features)
            results[k] = d
        results = sorted([(v,k) for k,v in results.items()])
        return results


def search_for_poke():
    index = pickle.load(open('images/pokedex/index.pkl', 'rb'))
    image = cv2.imread('images/pokedex/gameboy/query.png')
    #image = cv2.copyMakeBorder(image, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=(255,255,255))
    image = resize(image, height=56)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #ret, thresh = cv2.threshold(image.copy(), 245, 255, cv2.THRESH_BINARY_INV)
    thresh = cv2.adaptiveThreshold(image.copy(), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
#    canny = cv2.Canny(thresh, 90, 170)
#    cv2.imshow('canny', canny)
    outline = np.zeros(thresh.shape, dtype='uint8')
    (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    cv2.drawContours(outline, [cnts], -1, (255, 255, 255), -1)    
    desc = ZernikeMoments(21)
    queryFeatures = desc.describe(outline)
    searcher = Searcher(index)
    results = searcher.search(queryFeatures)
    result = results[0][1]
    print('that pokemon is: {}'.format(result.split('.')[0]))
    result_image = cv2.imread('images/pokedex/sprites/red-blue/'+result)
    cat_image = cv2.cvtColor(np.hstack([image, thresh, outline]), cv2.COLOR_GRAY2BGR)
    cv2.imshow('result image', np.hstack([cat_image, result_image]))




if __name__ == '__main__':
    search_for_poke()
    
    
    
    
    
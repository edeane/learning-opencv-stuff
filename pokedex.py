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
        url = 'https://img.pokemondb.net/artwork/{}.jpg'.format(parsedName)
        r = requests.get(url)
        if r.status_code != 200:
            print('[x] error downloading {}'.format(parsedName))
            continue
        print('[x] downloading {}'.format(parsedName))
        f = open('images/pokedex/{}.jpg'.format(name.lower()), 'wb')
        f.write(r.content)
        f.close()
        
def my_test():
    image = cv2.imread('images/pokedex/eevee.jpg')
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
    dataset_path = 'images/pokedex/'
    dataset_list = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')]
    
    for i, poke in enumerate(dataset_list):
        image = cv2.imread(dataset_path+poke)
        image = cv2.copyMakeBorder(image, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=(255,255,255))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #gray = cv2.GaussianBlur(gray, (5,5), 0)
        thresh = cv2.bitwise_not(gray)
        thresh[thresh>10] = 255
        thresh[thresh<10] = 0
        cv2.imshow('thresh', thresh)
        outline = np.zeros(thresh.shape, dtype='uint8')
        (_, cnts, _) = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
        cv2.drawContours(outline, [cnts], -1, (255, 255, 255), -1)
        cv2.imshow('a{}'.format(i), np.hstack([gray, thresh, outline]))
        moments = desc.describe(outline)
        index[poke] = moments
    
    pickle.dump(index, open('images/pokedex/index.pkl', 'wb'))







if __name__ == '__main__':
    create_index()
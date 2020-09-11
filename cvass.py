#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 15:48:03 2019

@author: allen
"""
import cv2
import numpy as np
import operator
import matplotlib.pyplot as plt



def reduce_resolution(image,factor):

    width = int(image.shape[1] * factor / 100)
    height = int(image.shape[0] * factor / 100)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    print('Resized Dimensions : ', resized.shape)
    return resized



def compute_frequent_values(padded_matrix,window_size,original_image_shape):
    
    result=np.zeros((original_image_shape[0],original_image_shape[1]))
    
    for x in range(original_image_shape[0]):
        for y in range(original_image_shape[1]):
            window=padded_matrix[x:x+(window_size),y:y+(window_size)]
            #print(window)
            histogram_dict={}
            for wx in range(window_size):
                for wy in range(window_size):
                
                    if window[wx][wy]!=-1:
                        if window[wx][wy] in histogram_dict.keys():
                            histogram_dict[window[wx][wy]]=histogram_dict[window[wx][wy]]+1
                        else:
                            histogram_dict[window[wx][wy]]=1
                    
            result[x][y]=  max(histogram_dict.items(), key=operator.itemgetter(1))[0] 
        
    return result



def compute_histogram(padded_matrix,task1,wind_size,original_image_shape):

    result = task1

    for x in range(original_image_shape[0]):
        for y in range(original_image_shape[1]):
            window=padded_matrix[x:x+wind_size,y:y+wind_size]
            #print(x,y)
            #print(window)
            hist, _ = np.histogram(window.ravel(), bins=range(266))

            max = np.argmax(hist)
            result[x][y]=max

    return result





def convert_to_uint8(matrix):
    matrix= np.array(matrix, dtype = np.uint8)
    return matrix

def convert_to_grayscale(original_image):
    b,g,r=cv2.split(original_image)
    rm=0.299*r
    gm=0.587*g
    bm=0.114*b
    I1=cv2.add(rm,gm)
    gray_matrix=cv2.add(I1,bm)
    for x in range(gray_matrix.shape[0]):
        for y in range(gray_matrix.shape[1]):
            gray_matrix[x][y]=round(gray_matrix[x][y])
    #print("converted_grey_image\n",gray_matrix,"\n")
    return gray_matrix
    
    
def display_image(image):
    cv2.imshow('image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
def padimage(image,window_size):
    x = np.pad(image , pad_width=int((window_size)/2), mode='constant', constant_values=-1)
    return x
    

def oil_paint_effect(paddedmatrix,original_image,window_size):

    b, g, r=cv2.split(original_image)
    b_oil = b
    g_oil = g
    r_oil = r
    b=padimage(b,windowsize)
    g=padimage(g,windowsize)
    r=padimage(r,windowsize)

    #print("shape of input matrix of oil paint",paddedmatrix.shape)
    for x in range(original_image.shape[0]):
        for y in range(original_image.shape[1]):
            window=paddedmatrix[x:x+(window_size),y:y+(window_size)]
            #print("window",window)
            histogram_dict={}
            sumg=0
            sumb=0
            sumr=0
            count=0
            for wx in range(window_size):
                for wy in range(window_size):

                    
                    if window[wx][wy]==window[(int(window_size/2))][(int(window_size/2))]:

                        sumb=b[x+wx][y+wy]+sumb
                        sumg=g[x+wx][y+wy]+sumg
                        sumr=r[x+wx][y+wy]+sumr
                        count=count+1

            sumg_avg=sumg/count
            sumb_avg=sumb/count
            sumr_avg=sumr/count
            b_oil[x][y]=sumb_avg
            g_oil[x][y]=sumg_avg
            r_oil[x][y]=sumr_avg
    oil_paint=cv2.merge((b_oil,g_oil,r_oil))

    return oil_paint

def save_image(savename,image):
    cv2.imwrite(savename+".jpg", image)

                        
                        
def oil_paint(paddedmatrix,original_image,window_size):
    b, g, r = cv2.split(original_image)
    b = padimage(b, windowsize)
    g = padimage(g, windowsize)
    r = padimage(r, windowsize)
    b_oil = b
    g_oil = g
    r_oil = r
    #print("shape of input matrix of oil paint", paddedmatrix.shape)
    for x in range(original_image.shape[0]):
        for y in range(original_image.shape[1]):
            window = paddedmatrix[x:x + (window_size), y:y + (window_size)]
            # print("window",window)
            histogram_dict = {}
            sumg = 0
            sumb = 0
            sumr = 0
            count = 0
            for wx in range(window_size):
                for wy in range(window_size):
                    points=[]

                    if window[wx][wy] == window[(int(window_size / 2))][(int(window_size / 2))]:
                        points.append((x+wx,y+wy))
                        sumb = b[x + wx][y + wy] + sumb
                        sumg = g[x + wx][y + wy] + sumg
                        sumr = r[x + wx][y + wy] + sumr
                        count = count + 1



            sumg_avg = sumg / count
            sumb_avg = sumb / count
            sumr_avg = sumr / count

            for p in points:


                b_oil[p[0]][p[1]] = sumb_avg
                g_oil[p[0]][p[1]] = sumg_avg
                r_oil[p[0]][p[1]] = sumr_avg
    oil_paint = cv2.merge((b_oil, g_oil, r_oil))

    return oil_paint
                        
                        
                        
                    
                    

    

    

original_image=cv2.imread('light_rail.jpg',1)
print("original image shape:",original_image.shape)
reducedimage=reduce_resolution(original_image,40)
windowsize=17
task2=np.zeros((reducedimage.shape[0],reducedimage.shape[1]))
task1=np.zeros((reducedimage.shape[0],reducedimage.shape[1]))
task1=convert_to_grayscale(reducedimage)
#print("shape after task1",task1)
#display_image(task1)
save_image("task1",task1)
task1padded=padimage(task1,windowsize)
#print("shape after task1 padded",task1padded.shape)
task2=compute_histogram(task1padded,task1,windowsize,reducedimage.shape)
#print("shape after computing frequent",task2.shape)
task2=convert_to_uint8(task2)
#display_image(task2)
save_image("task2",task2)
#print("shape after converting to uint8",task2.shape)
task2padded=padimage(task2,windowsize)
#print("shape after padding task2",task2padded.shape)

task3=oil_paint_effect(task2padded,reducedimage,windowsize)
task3=convert_to_uint8(task3)
save_image("task3",task3)
blue,green,red=cv2.split(task3)
#print("blue",blue)
#print("green",green)
#print("red",red)
display_image(task3)







        




        





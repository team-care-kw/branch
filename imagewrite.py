# -*- coding: utf-8 -*-
"""
Created on Mon May 11 14:48:58 2020

@author: Boombada
"""

import cv2
import datetime
from matplotlib import pyplot as plt
from time import gmtime, strftime

cap1 = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(2)


index = 1
while(cap1.isOpened() & cap2.isOpened()):
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if (ret1 & ret2):
        cv2.imshow('video1.', frame1)
        cv2.imshow('video2.', frame2)
        
        if cv2.waitKey(1) & 0xFF ==ord('q'):
              #  curTime = strftime("%M_%S",gmtime()
                cv2.imwrite('C:/Users/Boombada/Desktop/myPy/'+str(index)+ '_A.jpg',frame1)
                cv2.imwrite('C:/Users/Boombada/Desktop/myPy/'+str(index)+ '_B.jpg',frame2)
                index +=1 

        
    else:
        break
    
cap1.release()
cap2.release()
cv2.destroyAllWindows()




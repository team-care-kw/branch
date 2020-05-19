# -*- coding: utf-8 -*-
"""
Created on Tue May 12 21:56:34 2020

@author: playbang
"""

import os

import cv2
import argparse
import chainer
import numpy as np
from scipy.spatial import distance
from media_reader import VideoReader, get_filename_without_extension
from pose_detector import PoseDetector, draw_person_pose
from logging import basicConfig, getLogger, DEBUG


def Mapping3d(pose1, pose2, dist, foc):
  result = np.zeros((18,3))
  
  for i in range(18):
    if ((pose1[i][0] == 0 
      and pose1[i][1] == 0)
      or(pose2[i][0] == 0 
      and pose2[i][1] == 0)):  
      continue
  
    result[i][0] = dist * pose1[i][0] / (pose1[i][0]-pose2[i][0])
    
    result[i][1] = dist * pose1[i][1] / (pose1[i][0]-pose2[i][0])
    
    result[i][2] = dist * foc / (pose1[i][0] - pose2[i][0])
  
  return result

def GetNormalization(pose3D):
    
  dis = distance.euclidean(pose3D[14],pose3D[15])
  print("distance")
  print(dis)
  disArray = np.zeros(3)
  disArray = (dis, dis, dis)
  
  for i in range(2, 18):
     pose3D[i] = np.subtract(pose3D[i], pose3D[1])
     pose3D[i] = np.divide(pose3D[i], disArray)
  pose3D[0] = np.subtract(pose3D[0],pose3D[1])
  pose3D[0] = np.divide(pose3D[0], disArray)
  pose3D[1] = np.subtract(pose3D[1],pose3D[1])
  pose3D[1] = np.divide(pose3D[1], disArray)
    
  return pose3D

def OurTrainingInput(pose3D):
    cutoff = np.zeros((7, 3))
    cutoff[0] = pose3D[17] # 왼쪽귀
    cutoff[1] = pose3D[16] # 오른쪽귀
    cutoff[2] = pose3D[15] # 왼쪽 눈
    cutoff[3] = pose3D[14] # 오른쪽 눈
    cutoff[4] = pose3D[0] #코
    cutoff[5] = pose3D[5] #왼쪽 어깨
    cutoff[6] = pose3D[2] #오른쪽 어깨
    cutoff = cutoff.reshape(1, 21)
    return cutoff
  
def isItValid(pose2D):
    cutoff = np.zeros((7, 3))
    cutoff[0] = pose2D[17] # 왼쪽귀
    cutoff[1] = pose2D[16] # 오른쪽귀
    cutoff[2] = pose2D[15] # 왼쪽 눈
    cutoff[3] = pose2D[14] # 오른쪽 눈
    cutoff[4] = pose2D[0] #코
    cutoff[5] = pose2D[5] #왼쪽 어깨
    cutoff[6] = pose2D[2] #오른쪽 어깨
    zero_ = np.zeros(3)
    
    for i in range(7):
        if(cutoff[i][0] == 0 and cutoff[i][1] == 0):
            return False
    return True

chainer.using_config('enable_backprop', False)

if __name__ == '__main__':
    basicConfig(level=DEBUG)
    logger = getLogger(__name__)
  #  parser = argparse.ArgumentParser(description='Pose detector')
  #  parser.add_argument('--video', type=str, default='', help='video file path')
  #  parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
  #  args = parser.parse_args()

  #  if args.video == '':
  #      raise ValueError('Either --video has to be provided')

    chainer.config.enable_backprop = False
    chainer.config.train = False

    # load model
    pose_detector = PoseDetector("posenet", "models/coco_posenet.npz", device= 0)

    resultDir = "data/video_result/"
    if not os.path.exists(resultDir):
        os.makedirs(resultDir)


   # file_body_name = get_filename_without_extension(args.video)
   # file_append_name = '_result.mp4'

   # video = cv2.VideoWriter(resultDir + file_body_name + file_append_name, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30.0, (1280, 720))
   # video_provider = VideoReader(args.video)
   # idx = 0
   # read video
    cap = cv2.VideoCapture(0)#VideoReader(args.video)
    
    cap2 = cv2.VideoCapture(1)#VideoReader(args.video)
    idx = 0
    f = open("C:/Users/Boombada/Desktop/myPy/inputData.txt", 'a')
    f1 = open("C:/Users/Boombada/Desktop/myPy/Label.txt", 'a')
    
    index = 0
  #  for img in video_provider:
    while(cap.isOpened() & cap2.isOpened()):
        ret1, img = cap.read()
        ret2, img2 = cap2.read()
    
        poses, _ = pose_detector(img)
        poses2, _ = pose_detector(img2)
         
        res_img = cv2.addWeighted(img, 0.6, draw_person_pose(img, poses), 0.4, 0)
        res_img2 = cv2.addWeighted(img2, 0.6, draw_person_pose(img2, poses2), 0.4, 0)
        
      
        #logger.debug("type: {}".format(type(poses)))
        #logger.debug("shape: {}".format(poses.shape))
        logger.debug("A")
        logger.debug(poses)
      
        logger.debug(poses2)
   
        
        # cv2.imshow(file_body_name + '_result', res_img)
        if(ret1 & ret2):
            cv2.imshow('video', res_img)
            cv2.imshow('video2', res_img2)
            
            #if cv2.waitKey(1) & 0xFF ==ord('q'):
              #  curTime = strftime("%M_%S",gmtime()
            c = cv2.waitKey(1)

            if c == ord('t'): 
                
               
               
                arr1 = poses[0].reshape(18, 3)
                arr2 = poses2[0].reshape(18, 3)
                
                if(isItValid(arr1) == False):
                    print("pose 1 isn't valid")
                    continue
                if(isItValid(arr2) == False):
                    print("poses2 isn't valid")
                    continue
               
                
                #supervised = np.zeros(2)
                supervised = [[1, 0]]
                np.savetxt(f1,supervised,fmt='%d', delimiter = ',')
                
                cv2.imwrite('C:/Users/Boombada/Desktop/myPy/'+str(index)+ '_A.jpg',res_img)
                cv2.imwrite('C:/Users/Boombada/Desktop/myPy/'+str(index)+ '_B.jpg',res_img2)
                
                
                result = Mapping3d(arr1, arr2, 50, 400) 
                np.savetxt('C:/Users/Boombada/Desktop/myPy/'+ str(index) +'a.txt',arr1,fmt='%f', delimiter = ',')
                np.savetxt('C:/Users/Boombada/Desktop/myPy/'+ str(index) +'b.txt',arr2,fmt='%f', delimiter = ',')
                np.savetxt('C:/Users/Boombada/Desktop/myPy/'+ str(index) + '_result.txt',result,fmt='%f', delimiter = ',')
                Norm = GetNormalization(result)
                training_input = OurTrainingInput(Norm)
                np.savetxt(f,training_input,fmt='%f', delimiter = ',')
                index +=1 
                print("Print")
                
            elif c ==ord('q'):
                f.close()
                f1.close()
                cap.release()
                cap2.release()
                cv2.destroyAllWindows()
                break;
                
            elif c ==ord('f'):
                
               
                arr1 = poses[0].reshape(18, 3)
                arr2 = poses2[0].reshape(18, 3)
                
                if(isItValid(arr1) == False):
                    print("pose 1 isn't valid")
                    continue
                if(isItValid(arr2) == False):
                    print("poses2 isn't valid")
                    continue
                supervised = [[0, 1]]
                np.savetxt(f1,supervised,fmt='%d', delimiter = ',')
                
                
                cv2.imwrite('C:/Users/Boombada/Desktop/myPy/'+str(index)+ '_A.jpg',res_img)
                cv2.imwrite('C:/Users/Boombada/Desktop/myPy/'+str(index)+ '_B.jpg',res_img2)
                
                result = Mapping3d(arr1, arr2, 50, 400) 
                np.savetxt('C:/Users/Boombada/Desktop/myPy/'+ str(index) +'a.txt',arr1,fmt='%f', delimiter = ',')
                np.savetxt('C:/Users/Boombada/Desktop/myPy/'+ str(index) +'b.txt',arr2,fmt='%f', delimiter = ',')
                np.savetxt('C:/Users/Boombada/Desktop/myPy/'+ str(index) + '_result.txt',result,fmt='%f', delimiter = ',')
                Norm = GetNormalization(result)
                training_input = OurTrainingInput(Norm)
                np.savetxt(f,training_input,fmt='%f', delimiter = ',')
                index +=1 
                print("Print")
                
        # print('Saving file into {}{}{}{}'.format(resultDir, file_body_name, str(idx), file_append_name))
        # cv2.imwrite(resultDir + file_body_name + str(idx) + file_append_name, res_img)
        # idx += 1
      #  filehandler = logger.FileHandler('my.log')
      #  logger.addHandler(filehandler)
        
        cv2.waitKey(1)
        



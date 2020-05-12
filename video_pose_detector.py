import os

import cv2
import argparse
import chainer

from media_reader import VideoReader, get_filename_without_extension
from pose_detector import PoseDetector, draw_person_pose
from logging import basicConfig, getLogger, DEBUG

chainer.using_config('enable_backprop', False)

if __name__ == '__main__':
    basicConfig(level=DEBUG)
    logger = getLogger(__name__)
  #  parser = argparse.ArgumentParser(description='Pose detector')
    
    #parser.add_argument('--video', type=str, default='', help='video file path')
  #  parser.add_argument('--gpu', '-g', type=int, default= 0, help='GPU ID (negative value indicates CPU)')
  #  args = parser.parse_args()


#    if args.video == '':
#       raise ValueError('Either --video has to be provided')

    chainer.config.enable_backprop = False
    chainer.config.train = False

    cap = cv2.VideoCapture(1)
    
    print("videocapture")
    # load model
    pose_detector = PoseDetector("posenet", "models/coco_posenet.npz", device= 0)

#    resultDir = "data/video_result/"

 #   if not os.path.exists(resultDir):
  #      os.makedirs(resultDir)

#    file_body_name = get_filename_without_extension(args.video)
 #   file_append_name = '_result.mp4'

   # video = cv2.VideoWriter(resultDir + "test" + file_append_name, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30.0, (1280, 720))
    # read video

    
    cap = cv2.VideoCapture(1)#VideoReader(args.video)
    idx = 0
    cap2 = cv2.VideoCapture(2)#VideoReader(args.video)
    idx = 0
    #print("ssS")
    
   # while(cap.isOpened()):
    ret1, img = cap.read()
    ret2, img2 = cap2.read()
    #print("1")        
    #cv2.imshow('video1', img)
    poses, _ = pose_detector(img)
    poses2, _ = pose_detector(img2)
    #print("3")
    #cv2.imshow('video2', img)
    res_img = cv2.addWeighted(img, 0.6, draw_person_pose(img, poses), 0.4, 0)
    res_img2 = cv2.addWeighted(img2, 0.6, draw_person_pose(img2, poses2), 0.4, 0)
    logger.debug("type: {}".format(type(poses)))
    logger.debug("shape: {}".format(poses.shape))
    logger.debug(poses)
    # cv2.imshow(file_body_name + '_result', res_img)
   # video.write(res_img)
    if(ret1):
       cv2.imshow('video', res_img)
       cv2.imshow('video2', res_img2)
         #  print("2")
      #     if cv2.waitKey(1) & 0xFF ==ord('q'):
       #          break
       # else:
        #   break
       
      #  if(ret2):
           
            
       #     if cb2.waitKey(1) & 0xFF ==ord('q'):
        #        break
        
        # print('Saving file into {}{}{}{}'.format(resultDir, file_body_name, str(idx), file_append_name))
        # cv2.imwrite(resultDir + file_body_name + str(idx) + file_append_name, res_img)
        # idx += 1
        #cv2.waitKey(1)

    cv2.imwrite('img1', res_img)
    cv2.imwrite('img2', res_img2)
    
    cap.release()
    cap2.release()
    cv2.destroyAllWindows()

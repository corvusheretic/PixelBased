# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 14:41:21 2015

@author: kalyan
"""

import os
import os.path

import numpy as np
import cv2

import Pycluster

if __name__=='__main__':
    CharsDir  = "/home/kalyan/Python/PixelBased/SyntheticData";
    CharsList = os.listdir(CharsDir);
    CharsList.sort();
    
    for char in CharsList:
        AlphaDir = os.path.join(CharsDir, char);
        AlphaList = os.listdir(AlphaDir);
        AlphaList.sort();
        
        for alpha in AlphaList:
            fname = os.path.join(AlphaDir,alpha);
            wname = alpha.split('.');
            
            img  = cv2.imread(fname);
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);
            
#            sift   = cv2.xfeatures2d.SIFT_create();
#            kp,des = sift.detectAndCompute(gray,None);
#            
#            cv2.drawKeypoints(gray,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS);
#            cv2.imwrite(wname[0]+'_kp.png',img);
#                        
#            points = np.vstack([np.random.multivariate_normal(mean,
#                                                              0.03 * np.diag([1,1]),
#                                                                20) 
#                                for mean in [(1, 1), (2, 4), (3, 2)]]);
#            labels, error, nfound = Pycluster.kcluster(points, 3);
            print('Done');
            
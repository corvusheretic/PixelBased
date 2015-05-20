# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 08:33:19 2015

@author: kalyan
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 14:41:21 2015

@author: kalyan
"""

import os
import os.path
import shutil

import Image
import numpy as np
import cv2
#from pylab import *

from lib import SIFT

def toNum(s):
    try:
        return int(s)
    except ValueError:
        return float(s)


if __name__=='__main__':
    CharsDir  = "/home/kalyan/Python/PixelBased/bin/SyntheticData";
    CharsList = os.listdir(CharsDir);
    CharsList.sort();
    
    counter = 1;

    cwd     = os.path.dirname(os.path.realpath(__file__));
    
    DataDir = os.path.join(cwd, '../bin/SIFTData/');
    try:
        os.stat(DataDir);
    except:
        os.mkdir(DataDir);
    
    for char in CharsList:
        AlphaDir = os.path.join(CharsDir, char);
        AlphaList = os.listdir(AlphaDir);
        AlphaList.sort();
        
        alphaDir = os.path.join(DataDir, char);
        try:
            os.stat(alphaDir);
        except:
            os.mkdir(alphaDir);
            
        for alpha in AlphaList:                        
            wname = alpha.split('.');
            
            sift  = SIFT();
            fname = os.path.join(AlphaDir,alpha);
            cwd     = os.path.dirname(os.path.realpath(__file__));
            
            shutil.copy2(fname, cwd);
            TSDES  = wname[0]+'_temp.sift';
            SDES  = wname[0]+'.sift';
            
            img = np.array(Image.open(alpha).convert('L'));
            
            sift.dense(alpha,TSDES,128,1);
            fSiftDes = open(TSDES, 'r');
            SiftDes  = open(SDES, 'w');
            
            for line in fSiftDes:
                lline = line.split(' ');
                xCor,yCor = (toNum(lline[0])-1, toNum(lline[1])-1);
                
                if(img[yCor,xCor] == 255):
                    lline.insert(0,str(0));
                else:
                    lline.insert(0,str(counter));
                
                lline = " ".join(lline);
                
                SiftDes.write(lline);
            
            counter += 1;
            
            shutil.copy2(SDES, alphaDir);
            
            os.remove(alpha);
            os.remove(TSDES);
            os.remove(SDES);
            
            print("================ %d. SIFT done for alphabet: %s ================\n" % (counter,alpha, ));
            
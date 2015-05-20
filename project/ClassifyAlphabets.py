# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 11:32:02 2015

@author: kalyan
"""

import os
import os.path
import shutil
import operator

from random import shuffle
import pickle

from PIL import Image
import numpy as np
import cv2

from lib import SIFT
from lib import BayesClassifier
#from src import CreateAlphabetImage
#from src import GetbBox

import matplotlib.pyplot as plt

from HTML import table

from oct2py import octave

import genJS

BINARIZE  = 0;
NORMALIZE = 1;
alphaBkg = {'A':1,  'B':2,  'C':3,  'D':4,
            'E':5,  'F':6,  'G':7,  'H':8,
            'I':9,  'J':10, 'K':11, 'L':12,
            'M':13, 'N':14, 'O':15, 'P':16,
            'Q':17, 'R':18, 'S':19, 'T':20,
            'U':21, 'V':22, 'W':23, 'X':24,
            'Y':25, 'Z':26, 'a':27, 'b':28,
            'c':29, 'd':30, 'e':31, 'f':32,
            'g':33, 'h':34, 'i':35, 'j':36,
            'k':37, 'l':38, 'm':39, 'n':40,
            'o':41, 'p':42, 'q':43, 'r':44,
            's':45, 't':46, 'u':47, 'v':48,
            'w':49, 'x':50, 'y':51, 'z':52};
            
def toNum(s):
    try:
        return int(s)
    except ValueError:
        return float(s)

def separateTrainTest(ImgDir,nTrainData,nTestData):
    cwd     = os.path.dirname(os.path.realpath(__file__));
    
    TrainDataDir = os.path.join(cwd, '../nbin/train');
    TestDataDir  = os.path.join(cwd, '../nbin/test');
    
    try:
        os.stat(TrainDataDir);
    except:
        os.mkdir(TrainDataDir);
    
    try:
        os.stat(TestDataDir);
    except:
        os.mkdir(TestDataDir);
    
    shutil.rmtree(TrainDataDir);
    shutil.rmtree(TestDataDir);
    os.mkdir(TrainDataDir);
    os.mkdir(TestDataDir);

    CharsList = os.listdir(ImgDir);
    CharsList.sort();
    
    for char in CharsList:
        AlphaDir = os.path.join(CharsDir, char);
        AlphaList = os.listdir(AlphaDir);
        shuffle(AlphaList);
        
        counter = 0;
        
        for alpha in AlphaList:
            if(counter < nTrainData+nTestData):
                fname = os.path.join(AlphaDir,alpha);
                if(counter < nTrainData):
                    shutil.copy2(fname, TrainDataDir);
                else:
                    shutil.copy2(fname, TestDataDir);
            else:
                break;
            
            counter += 1;


def imgReSize(alpha,ReSizeSq):
    cwd     = os.path.dirname(os.path.realpath(__file__));
    fname = os.path.join(CharsDir,alpha);
    shutil.copy2(fname, cwd);
                
    img = Image.open(alpha).convert('L');    
    im = np.array(img);
    
    # Compute the magnification to maintain the aspect ratio
    magWd = ReSizeSq[0]/float(im.shape[1]);
    magHi = ReSizeSq[1]/float(im.shape[0]);
    
    mag = min(magWd,magHi);
    SizeSq = (int(np.floor(im.shape[1]*mag) +1), int(np.floor(im.shape[0]*mag) +1));

    img = img.resize(SizeSq, Image.ANTIALIAS);
    img = np.array(img);
    
    if(len(img.shape)==3):
        img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY);
    
    if(BINARIZE):
        ret2,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU);
    
    return (img, SizeSq);


def getDenseSiftData(img,binSize,stepSize,offset,ch,rPixelValue):
    
    loc,des = octave.vl_dsift(img.astype(np.float32),'size',binSize,'step',stepSize,'floatdescriptors');
    
    loc      = (loc.astype(np.int)).T.tolist();
    features = des.T.tolist();
    
    labels    = [];
    intensity = [];


    for (y,x) in loc:
        x -= 1; y -= 1;
        pVal = img[x,y];
        if(rPixelValue):
            intensity.append(str(pVal));
        
        if(img[x,y] == 255):
            labels.append(str(alphaBkg[ch[0]]));
        else:
            labels.append(ch[0]);
        
    return features,loc,labels,intensity;

def getImgMeanCov(data,labels):
    
    xLabels = {};
    xData   = [];
    
    cnt=0;
    for ch in labels:
        val = np.array(data[cnt]);
        if ch in xLabels:
            idx = xLabels[ch];
            classMat   = xData[idx];
            classMat   = np.vstack((classMat,val));
            xData[idx] = classMat;
        else:
            idx = len(xLabels);
            xLabels[ch] = idx;
            xData.append(val);
        cnt += 1;
    
    mean = [0]*len(xLabels);
    cov  = [0]*len(xLabels);
    n    = [0]*len(xLabels);
    
    for ch in xLabels:
        idx = xLabels[ch];
        mean[idx] = np.mean(xData[idx],0);
        cov[idx]  = np.cov(xData[idx].T);
        n[idx]    = float(xData[idx].shape[0]);
    
    xLabels = sorted(xLabels.items(), key=operator.itemgetter(0));
    
    return(xLabels,mean,cov,n);

    
def paintClassification(iim,loc,offset,res,labels,name,step,usedRes,magnify):
    
    cwd   = os.path.dirname(os.path.realpath(__file__));
    ClassifyDir = os.path.join(cwd, '../nbin/AlphabetData/heatmap');
    
    img = cv2.cvtColor(iim,cv2.COLOR_GRAY2RGB);
    
    imgR = img[:,:,0];
    imgG = img[:,:,1];
    imgB = img[:,:,2];
    
    labCnt = 0;

    for yC,xC in loc:
        if(res[labCnt] != labels[labCnt]):
            # Case: Mismatch of labels
            if(iim[xC,yC]==255):
                imgR[xC:xC+step,yC:yC+step] = 0;
                imgG[xC:xC+step,yC:yC+step] = 0;
                imgB[xC:xC+step,yC:yC+step] = 225;
            else:
                imgR[xC:xC+step,yC:yC+step] = 225;
                imgG[xC:xC+step,yC:yC+step] = 0;
                imgB[xC:xC+step,yC:yC+step] = 0;
        else:
            # Case: Labels matched
            if(iim[xC,yC]==255):
                imgR[xC:xC+step,yC:yC+step] = 225;
                imgG[xC:xC+step,yC:yC+step] = 225;
                imgB[xC:xC+step,yC:yC+step] = 0;
            else:
                imgR[xC:xC+step,yC:yC+step] = 0;
                imgG[xC:xC+step,yC:yC+step] = 225;
                imgB[xC:xC+step,yC:yC+step] = 0;
        labCnt  += 1;
    
    im = np.zeros((img.shape[0]-2*offset,img.shape[1]-2*offset,3),np.uint8);
    im[:,:,0] = imgR[offset:imgR.shape[0]-offset, offset:imgR.shape[1]-offset];
    im[:,:,1] = imgG[offset:imgG.shape[0]-offset, offset:imgG.shape[1]-offset];
    im[:,:,2] = imgB[offset:imgB.shape[0]-offset, offset:imgB.shape[1]-offset];
        
    
    img = Image.fromarray(im);
    
    disRes = (usedRes[0]*magnify,usedRes[1]*magnify);
    
    img = img.resize(disRes, Image.ANTIALIAS);            
    outImgName = os.path.join(ClassifyDir,name);
    img.save(outImgName);

def debugPrint(debugPDF,estPDF,estLab,lab,xC,yC,sz):
    
    pdfStr=[];
    
    for nc in range(len(debugPDF)):
        cPDF = debugPDF[nc];
        pdfStr.append(int(cPDF[yC,xC]));
    
    print('Debug PDF \n');
    print(pdfStr);
   
    pdfStr=[];

    cPDF = estPDF[xC*sz[0]+yC,:];
    for nc in range(cPDF.shape[0]):
        pdfStr.append(int(cPDF[nc]));
    
    print('Est PDF \n');
    print(pdfStr);
    
    print('Est Lab \n');
    print(estLab[xC*sz[0]+yC]);
    print('True Lab \n');
    print(lab[xC*sz[0]+yC]);
    

def debugLabels(im,offset,cords,lab):
    
    img = (im==255);
    img = img.astype(np.uint8);
    
    ximg = np.ones(img.shape,np.int8);
    
    cnt = 0;
    for ch in lab:
        (y,x) = cords[cnt];
        x -=1; y -=1;
        if(ch != '0'):
            ximg[x,y] = 0;
        cnt += 1;
    
    eimg = (255*(np.logical_xor(ximg,img).astype(np.uint8))).astype(np.uint8);
    
    cv2.imwrite('binImg.png',(255*img).astype(np.uint8));
    cv2.imwrite('label.png',(255*ximg).astype(np.uint8));
    cv2.imwrite('err_map.png',eimg);
        
if __name__=='__main__':
    
    CharsDir    = "/home/kalyan/Python/PixelBased/nbin/SyntheticData";
    ScaleSpace  = 15;
    TrainStepSz = 1;
    TestStepSz  = 1;
    
    #ReSizeSq          = (100,100);
    #ReSizeSq     = (50,50);
    ReSizeSq     = (70,50);
    MarkMagnify  = 40; #40*(50,50)=(2000,2000)
    PaintMagnify = 10;#10*(50,50)=(500,500);
    
    NUM_TRAIN_DATA   = 40;
    NUM_TEST_DATA    = 2;
    
    GENERATE         = 1;
    SIFT_DATA        = 0;
    DSIFT_DATA       = not(SIFT_DATA);
    BAYES_CLASSIFY   = 1;
    READ_PIXEL_VALUE = 0;
    
    START_BINSZ = 6;
    END_BINSZ   = 6;
    STEP_BINSZ  = 2;
    
    # :: INITIALIZATION ::
    #       >> Separate training and test data
    #       >> setup the intermediate and output directory structure
    cwd     = os.path.dirname(os.path.realpath(__file__));
    octave.addpath('/home/kalyan/VlFeat/vlfeat-0.9.20/toolbox');
    octave.vl_setup();
    
    if( GENERATE and 0 ):
        separateTrainTest(CharsDir,NUM_TRAIN_DATA,NUM_TEST_DATA);

        SaveDir = os.path.join(cwd, '../nbin/AlphabetData/pdf');
        try:
            os.stat(SaveDir);
        except:
            os.mkdir(SaveDir);
        
        shutil.rmtree(SaveDir);
        os.mkdir(SaveDir);
        
        ClassifyDir = os.path.join(cwd, '../nbin/AlphabetData/heatmap');
        try:
            os.stat(ClassifyDir);
        except:
            os.mkdir(ClassifyDir);
        
        shutil.rmtree(ClassifyDir);
        os.mkdir(ClassifyDir);
    
    
    
    TRAIN = 0;
    for TrainBinSz in range(START_BINSZ,END_BINSZ+STEP_BINSZ,STEP_BINSZ):
        bc = BayesClassifier();
        print("\n\n////////// PROCESS:: bin-size %d //////////\n" % (TrainBinSz, )),;
        
        TestBinSz = TrainBinSz;
        
        # :: TRAIN LDA ::
        #       >> Per file identify the labels
        #       >> Update the classes instances, mean, var iteratively for 
        #           all class per file.
        
        if(TRAIN):
            CharsDir  = os.path.join(cwd, '../nbin/');
            CharsDir  = os.path.join(CharsDir, 'train');
            CharsList = os.listdir(CharsDir);
            CharsList.sort();
            
            counter = 1;
            CURR_ALPHA='';
            for alpha in CharsList:
                wname = alpha.split('.');
                ch    = alpha.split('_')[0];
                
                im, SizeSq = imgReSize(alpha,ReSizeSq);
                
                offset = int(1.5*TrainBinSz);
                
                img = 255*np.ones((im.shape[0]+2*offset,im.shape[1]+2*offset),np.int8);
                img = img.astype(np.uint8);
                img[offset:im.shape[0]+offset, offset:im.shape[1]+offset] = im;
                os.remove(alpha);
                
                if(GENERATE & DSIFT_DATA):
                    feat,_,lab,_ = getDenseSiftData(img, TrainBinSz, TrainStepSz, offset, ch, READ_PIXEL_VALUE);
                
                if(CURR_ALPHA == ch):
                    counter += 1;
                    print("%d. %s," % (counter,alpha,)),;
                else:
                    CURR_ALPHA = ch;
                    counter = 1;
                    print("\n\n===== BEGIN:: dSIFT training for alphabet %s =====" % (ch, ));
                    print("%d. %s," % (counter,alpha,)),;
                    
                #print("============ %d. DONE:: SIFT for %s ============\n" % (counter,alpha, ));
                #counter += 1;
                
                xLabels,imgMean,imgCov,nSamp = getImgMeanCov(feat,lab);
                # train LDA
                bc.train(xLabels,imgMean,imgCov,nSamp);

        # :: TRAIN LDA ::
        #       >> Per file identify the labels
        #       >> Update the classes instances, mean, var iteratively for 
        #           all class per file.
            bc.saveState();
        
        if(not(TRAIN)):
            bc.loadState();
        
        CharsDir  = os.path.join(cwd, '../nbin/');
        CharsDir  = os.path.join(CharsDir, 'test');
        CharsList = os.listdir(CharsDir);
        CharsList.sort();
        
        counter = 1;
        CURR_ALPHA='';
        for alpha in CharsList:
            wname = alpha.split('.');
            ch    = alpha.split('_')[0];
            
            im, SizeSq = imgReSize(alpha,ReSizeSq);
            
            offset = int(1.5*TrainBinSz);
            
            img = 255*np.ones((im.shape[0]+2*offset,im.shape[1]+2*offset),np.int8);
            img = img.astype(np.uint8);
            img[offset:im.shape[0]+offset, offset:im.shape[1]+offset] = im;
            
            if(GENERATE & DSIFT_DATA):
                feat,cords,lab,_ = getDenseSiftData(img, TestBinSz, TestStepSz, offset, ch, READ_PIXEL_VALUE);
            
            #debugLabels(img,offset,cords,lab);
            
            # test LDA
            estLab, estProb = bc.classify(np.array(feat));
            
            acc = sum(1.0*(np.array(estLab) == np.array(lab))) / len(lab);
            #print("============ %d. DONE:: Classification of %s with Accuracy: %f ============\n" % (counter,alpha,acc, ));
            
            if(CURR_ALPHA == ch):
                counter += 1;
                print("%d. %s with accuracy: %f," % (counter,alpha,acc,)),;
            else:
                CURR_ALPHA = ch;
                counter = 1;
                print("\n\n===== BEGIN:: Classification for alphabet %s =====" % (ch, ));
                print("%d. %s with accuracy: %f," % (counter,alpha,acc,)),;
                    
            name = wname[0]+'_'+str(TestBinSz)+'.'+wname[1];
            
            paintClassification(img.astype(np.uint8),cords,offset,estLab,lab,name,TestStepSz,SizeSq,10);
            
            if 0:
                genJS.heatmapBkgLayer(alpha,im.shape,4);
                #print('Bkg Layer ready.\n');
                debugPDF = genJS.classHeatmap(alpha,im.shape,TestBinSz,estProb,4);
                #genJS.classHeatmap(alpha,TestBinSz,estProb);
        
            os.remove(alpha);
            #debugPrintPDF(debugPDF,1,1);
            #debugPrintEstPDF(estProb,1,1,im.shape);
            #debugPrint(debugPDF,estProb,estLab,lab,1,1,im.shape);
            print('');
            
        
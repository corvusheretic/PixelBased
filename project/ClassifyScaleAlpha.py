# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 11:32:02 2015

@author: kalyan
"""

import os
import os.path
import shutil

from random import shuffle
import pickle

from PIL import Image, ImageDraw
import numpy as np
import cv2

from lib import SIFT
from lib import PCA
from lib import BayesClassifier
from src import CreateAlphabetImage
from src import GetbBox

import matplotlib.pyplot as plt

from HTML import table

from oct2py import octave

def toNum(s):
    try:
        return int(s)
    except ValueError:
        return float(s)

def separateTrainTest(ImgDir,nTrainData,nTestData):
    cwd     = os.path.dirname(os.path.realpath(__file__));
    
    TrainDataDir = os.path.join(cwd, '../bin/train');
    TestDataDir  = os.path.join(cwd, '../bin/test');
    
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
        

def getSiftData(path,nFeat,sampStep,reSz):
    cwd     = os.path.dirname(os.path.realpath(__file__));
    CharsDir  = os.path.join(cwd, '../bin/');
    CharsDir  = os.path.join(CharsDir, path);
    CharsList = os.listdir(CharsDir);
    CharsList.sort();
    
    counter = 1;
    
    SaveDir = os.path.join(cwd, '../bin/AlphabetData/');

    try:
        os.stat(SaveDir);
    except:
        os.mkdir(SaveDir);
    
    DataDir = os.path.join(SaveDir, path);
    try:
        os.stat(DataDir);
    except:
        os.mkdir(DataDir);
    
    shutil.rmtree(DataDir);
    os.mkdir(DataDir);
    
    FileName = path+'.set';
    
    for alpha in CharsList:                        
        wname = alpha.split('.');
        ch    = alpha.split('_');
        
        sift  = SIFT();
        fname = os.path.join(CharsDir,alpha);
        
        shutil.copy2(fname, cwd);
        
        TSDES  = wname[0]+'_temp.sift';
        SDES  = wname[0]+'.sift';
        
        img = Image.open(alpha).convert('L');
        img = img.resize(reSz, Image.ANTIALIAS);
        img = np.array(img);
        #cv2.imwrite(alpha,img);
        
        sift.dense(alpha,TSDES,nFeat,sampStep,resize=reSz);
        fSiftDes = open(TSDES, 'r');
        SiftDes  = open(SDES, 'w');
        
        with open(TSDES) as fSiftDes:
            for line in fSiftDes:
                lline = line.split(' ');
                xCor,yCor = (toNum(lline[0]), toNum(lline[1]));
                
                if( not((0 <= xCor) and (xCor <= reSz[0]-sampStep)) ):
                    print('ERROR: Position');
                if( not((0 <= yCor) and (yCor <= reSz[1]-sampStep)) ):
                    print('ERROR: Position');
                    
                if(np.min(img[yCor:yCor+sampStep,xCor:xCor+sampStep]) == 255):
                    lline.insert(0,str(0));
                else:
                    lline.insert(0,ch[0]);
                
                lline = " ".join(lline);
                
                SiftDes.write(lline);
                SiftDes.flush();
        
        fSiftDes.close();
        SiftDes.close();
        shutil.copy2(SDES, DataDir);
        
        
        with open(FileName, 'wb') as ff:
            pickle.dump(CharsList, ff);
        
        shutil.copy2(FileName, SaveDir);
        
        os.remove(alpha);
        os.remove(TSDES);
        os.remove(SDES);
        os.remove(FileName);

        print("================ %d. SIFT done for alphabet: %s ================\n" % (counter,alpha, ));
        counter += 1;

def getDenseSiftData(path,binSize,stepSize,reSz):
    cwd       = os.path.dirname(os.path.realpath(__file__));
    OctDir    = os.path.join(cwd, './octave/');
    octave.addpath(OctDir);
    
    CharsDir  = os.path.join(cwd, '../bin/');
    CharsDir  = os.path.join(CharsDir, path);
    CharsList = os.listdir(CharsDir);
    CharsList.sort();
    
    counter = 1;
    
    SaveDir = os.path.join(cwd, '../bin/AlphabetData/');

    try:
        os.stat(SaveDir);
    except:
        os.mkdir(SaveDir);
    
    DataDir = os.path.join(SaveDir, path);
    try:
        os.stat(DataDir);
    except:
        os.mkdir(DataDir);
    
    shutil.rmtree(DataDir);
    os.mkdir(DataDir);
    
    FileName = path+'.set';
    
    for alpha in CharsList:                        
        wname = alpha.split('.');
        ch    = alpha.split('_');
        
        fname = os.path.join(CharsDir,alpha);
        
        shutil.copy2(fname, cwd);
        
        SDES  = wname[0]+'.dsift';
        SiftDes  = open(SDES, 'w');
        
        img = Image.open(alpha).convert('L');
        img = img.resize(reSz, Image.ANTIALIAS);
        img = np.array(img);
        #cv2.imwrite(alpha,img);
        
        #loc,des = octave.dsift(img,'Size',binSize,'Step',stepSize,1.5);
        loc,des = octave.dsift(img,binSize,stepSize,1.5);
        loc     = loc.astype(np.int);
        feat    = (np.vstack((loc,des))).T;
        feat    = feat.tolist();
        
        for item in feat:
            xCor,yCor = item[:2];
            if( not((0 <= xCor) and (xCor <= reSz[0]-binSize)) ):
                print('ERROR: Position');
            if( not((0 <= yCor) and (yCor <= reSz[1]-binSize)) ):
                print('ERROR: Position');
            
            llist= ' '.join(str(x) for x in item);
            
            if(np.min(img[yCor:yCor+binSize,xCor:xCor+binSize]) == 255):
                llist = str(0) + ' ' + llist;
            else:
                llist =  ch[0] + ' ' + llist;
            
            SiftDes.write("%s\n" % llist);
            SiftDes.flush();
                
        SiftDes.close();
        shutil.copy2(SDES, DataDir);
        
        
        with open(FileName, 'wb') as ff:
            pickle.dump(CharsList, ff);
        
        shutil.copy2(FileName, SaveDir);
        
        os.remove(alpha);
        os.remove(SDES);
        os.remove(FileName);

        print("================ %d. Dense SIFT done for alphabet: %s ================\n" % (counter,alpha, ));
        counter += 1;

def readFeaturesLabels(path,reSz,rPixelValue,isDense):

    cwd   = os.path.dirname(os.path.realpath(__file__));
    DataDir = os.path.join(cwd, '../bin/AlphabetData/');
    
    DataDir = os.path.join(DataDir, path);
    
    # create list of all files ending in .sift/.dsift
    if(isDense):
        featlist = [os.path.join(DataDir,f) for f in os.listdir(DataDir) if f.endswith('.dsift')];
    else:
        featlist = [os.path.join(DataDir,f) for f in os.listdir(DataDir) if f.endswith('.sift')];
    featlist.sort();
    
    # read the features
    features = [];
    labels   = [];
    xyCords  = [];
    
    sift  = SIFT();
    
    DataDir = os.path.join(cwd, '../bin/');    
    DataDir = os.path.join(DataDir, path);
    
    for featfile in featlist:
        lab,loc,des = sift.readExtFeatures(featfile,isDense);
        if(rPixelValue):
            fn = featfile.split('/');
            fn = fn[-1];
            fn = fn.split('.');
            fname = fn[0]+'.png';
            fname = os.path.join(DataDir,fname);
            img = Image.open(fname).convert('L');
            img = img.resize(reSz, Image.ANTIALIAS);
            img = np.array(img);
            
            if(len(img.shape)==3):
                img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY);
            
            cnt =0;
            
            if(len(des) == len(loc)):
                for (x,y,_,_) in loc:
                    pVal = img[x,y];
                    des[cnt].insert(0,str(pVal));
                    cnt += 1;
            else:
                print("ERROR: Size mismatch.\n");
            
        features +=des;
        labels   +=lab;
        xyCords  +=loc;
        
        fileName = featfile.split('/')[-1];
        fileName = fileName.split('.');
    
    features = np.array(features);
    features = features.astype(np.int);
    # create labels
    labels   = np.array(labels);
    xyCords  = np.array(xyCords)[:,:2];
    
    return features,labels,xyCords;

def printConfusion(res,labels,classnames,funName):
    
    n = len(classnames);

    # confusion matrix
    class_ind = dict([(classnames[i],i) for i in range(n)]);

    confuse = np.zeros((n,n));
    for i in range(len(labels)):
        confuse[class_ind[res[i]],class_ind[labels[i]]] += 1;

    #print 'Confusion matrix for'    
    #print classnames;
    #print confuse;
    
    classnames_L = classnames.tolist();
    
    confuse = confuse.astype(np.int);
    confuse_L = confuse.tolist();
    class_confuse_L = [];
    
    for i in range(len(confuse_L)):
        ccn = confuse_L[i];
        ccn.insert(0,classnames_L[i]);
        class_confuse_L.append(ccn);
    
    classnames_L.insert(0,' ');
    
    c_styles     = ['']*len(classnames_L);
    c_styles[0]  = 'font-size: small';
    #c_styles[-1] = 'background-color:yellow';
    
    htmlcode = table(class_confuse_L,
        header_row = classnames_L,
        col_width=['']*len(classnames_L),
        col_align=['center']*len(classnames_L),
        col_styles=c_styles);
    
    
    f = open(funName+'.html', 'w')
    f.write(htmlcode + '<p>\n')
    #print htmlcode
    #print '-'*79
    f.close();
    
    if(0):
        f = open(funName+'.txt','w');
        np.savetxt(f,classnames,fmt='%c',newline=' ');
        f.write('\n');
        f.close();
        
        f = open(funName+'.txt','a');
        np.savetxt(f,confuse,fmt='%d',delimiter=' ');
        f.close();
    
    # Turn interactive plotting off
    plt.ioff();
    
    fig, axarr = plt.subplots(nrows=2, ncols=1);
    
    axarr[0].pcolor(confuse, cmap='Blues');
    axarr[0].set_title('Confusion Matrix');
    
    
    axarr[1].pcolor(np.log(confuse+1), cmap='Blues');
    axarr[1].set_title('log(Confusion Matrix)');
    
    #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    #fig.colorbar(fig,cax=cbar_ax);
    fig.savefig(funName+'.png');
    plt.close(fig);


def markClassification(res,name,usedRes,disRes):
    
    cwd   = os.path.dirname(os.path.realpath(__file__));
    SaveDir = os.path.join(cwd, '../bin/AlphabetData/');
    
    ClassifyDir = os.path.join(cwd, '../bin/AlphabetData/classify');
    try:
        os.stat(ClassifyDir);
    except:
        os.mkdir(ClassifyDir);
    
    shutil.rmtree(ClassifyDir);
    os.mkdir(ClassifyDir);
        
    FileName = name+'.set';
    FileName = os.path.join(SaveDir, FileName);    

    with open(FileName, 'rb') as ff:
        CharsList = pickle.load(ff);
    
    SaveDir = os.path.join(SaveDir, name);
    
    # create list of all files ending in .dsift
    featlist = [os.path.join(SaveDir,f) for f in os.listdir(SaveDir) if f.endswith('.sift')];
    featlist.sort();
    
    DataDir  = os.path.join(cwd, '../bin');
    DataDir  = os.path.join(DataDir, name);
          
    sift  = SIFT();
    counter = 0;
    labCnt  = 0;
    if(len(CharsList) == len(featlist)):
        for featfile in featlist:
            _,loc,_ = sift.readExtFeatures(featfile);
    
            alpha = CharsList[counter];
            fname = os.path.join(DataDir,alpha);
            
            shutil.copy2(fname, cwd);
            
            img  = Image.open(alpha);#.convert('L');
            img = img.resize(usedRes, Image.ANTIALIAS);
            draw = ImageDraw.Draw(img);
            im   = np.array(img);
            xsz, ysz = im.shape;
            
            loc = np.array(loc).astype(np.float);
            xCs = np.unique(loc[:,0]);
            yCs = np.unique(loc[:,1]);
            
            for xC in xCs:
                if( xC > 0 ):
                    draw.line((xC,0, xC,ysz), fill=128);
            for yC in yCs:
                if( yC > 0 ):
                    draw.line((0,yC, xsz,yC), fill=128);
            
            img = img.resize(disRes, Image.ANTIALIAS);
            #img.save('test.png');
            img = np.array(img);
            
            if(len(img.shape)==2):
                img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB);
            
            imgR = img[:,:,0];
            imgG = img[:,:,1];
            imgB = img[:,:,2];

            BImgSz = int(0.5*(np.min(disRes)/len(xCs)));
            colorMap = (0,0,255);
            fontName = '/home/kalyan/Python/PixelBased/bin/Fonts/Mixed/Aller_Lt.ttf'
            fontsize = 20;
            offset   = 1;
            
            for yC in yCs:
                for xC in xCs:
                    BImgxPos = xC*disRes[0]/usedRes[0] + offset*disRes[0]/usedRes[0];
                    BImgyPos = yC*disRes[1]/usedRes[1] + offset*disRes[1]/usedRes[1];
                    #pos      = (BImgxPos, BImgyPos);
                    txt      = res[labCnt];
                    labCnt  += 1;
                    
                    alphaImage = Image.new("RGB", (BImgSz,BImgSz), (255,255,255));
                    CreateAlphabetImage(alphaImage, fontName, fontsize, txt, (5,5), colorMap);
                    #alphaImage.save('alpha.png');
                    alphaImage = np.array(alphaImage);
                                        
                    aimgB = alphaImage[:,:,0];
                    aimgG = alphaImage[:,:,1];
                    aimgR = alphaImage[:,:,2];
                    
                    np_img     = cv2.cvtColor(alphaImage,cv2.COLOR_BGR2GRAY);
                    ret,np_img = cv2.threshold(np_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU);
            
                    (xmin,ymin,xmax,ymax) = GetbBox(np_img); # or aimgG will do
                    # We need these pixels values to be zero for alphabet
                    xmin = max(0,xmin-2);
                    ymin = max(0,ymin-2);
                    xmax = min(BImgSz,xmax+2);
                    ymax = min(BImgSz,ymax+2);
                    
                    aimgR = aimgR[xmin:xmax,ymin:ymax];
                    aimgG = aimgG[xmin:xmax,ymin:ymax];
                    aimgB = aimgB[xmin:xmax,ymin:ymax];
                    
                    imgR[BImgxPos:BImgxPos+np.size(aimgR,0), 
                         BImgyPos:BImgyPos+np.size(aimgR,1)] = aimgR;
                    imgG[BImgxPos:BImgxPos+np.size(aimgG,0), 
                         BImgyPos:BImgyPos+np.size(aimgG,1)] = aimgG;
                    imgB[BImgxPos:BImgxPos+np.size(aimgB,0), 
                         BImgyPos:BImgyPos+np.size(aimgB,1)] = aimgB;
            
            img[:,:,0] = imgR;
            img[:,:,1] = imgG;
            img[:,:,2] = imgB;
            
            outImgName = os.path.join(ClassifyDir,alpha);
            cv2.imwrite(outImgName,img);            
            
            counter += 1;
            os.remove(alpha);
    
def paintClassification(res,labels,name,step,usedRes,disRes,isDense):
    
    cwd   = os.path.dirname(os.path.realpath(__file__));
    SaveDir = os.path.join(cwd, '../bin/AlphabetData/');
    
    ClassifyDir = os.path.join(cwd, '../bin/AlphabetData/paint');
    try:
        os.stat(ClassifyDir);
    except:
        os.mkdir(ClassifyDir);
    
    shutil.rmtree(ClassifyDir);
    os.mkdir(ClassifyDir);
        
    FileName = name+'.set';
    FileName = os.path.join(SaveDir, FileName);    

    with open(FileName, 'rb') as ff:
        CharsList = pickle.load(ff);
    
    SaveDir = os.path.join(SaveDir, name);
    
    # create list of all files ending in .dsift
    if(isDense):
        featlist = [os.path.join(SaveDir,f) for f in os.listdir(SaveDir) if f.endswith('.dsift')];
    else:
        featlist = [os.path.join(SaveDir,f) for f in os.listdir(SaveDir) if f.endswith('.sift')];
    featlist.sort();
    
    DataDir  = os.path.join(cwd, '../bin');
    DataDir  = os.path.join(DataDir, name);
          
    sift  = SIFT();
    counter = 0;
    labCnt  = 0;
    if(len(CharsList) == len(featlist)):
        for featfile in featlist:
            _,loc,_ = sift.readExtFeatures(featfile,isDense);
    
            alpha = CharsList[counter];
            fname = os.path.join(DataDir,alpha);
            
            shutil.copy2(fname, cwd);
            
            img  = Image.open(alpha);#.convert('L');
            img  = img.resize(usedRes, Image.ANTIALIAS);
            im   = np.array(img);
            
            if(len(im.shape)==2):
                img = cv2.cvtColor(im,cv2.COLOR_GRAY2RGB);
            else:
                if(len(im.shape)==3):
                    img   = np.array(img);
                    im    = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY);
                else:
                    print("ERROR: Wrong image format.");
            
            xsz, ysz = im.shape;
            
            imgR = img[:,:,0];
            imgG = img[:,:,1];
            imgB = img[:,:,2];
            
            loc = np.array(loc).astype(np.float);
            xCs = np.unique(loc[:,0]);
            yCs = np.unique(loc[:,1]);            
            
            for yC in yCs:
                for xC in xCs:
                    if(res[labCnt] != labels[labCnt]):
                        if(im[xC,yC]==255):
                            imgR[xC:xC+step,yC:yC+step] = 64;
                            imgG[xC:xC+step,yC:yC+step] = 64;
                            imgB[xC:xC+step,yC:yC+step] = 128;
                        else:
                            imgR[xC:xC+step,yC:yC+step] = 128;
                            imgG[xC:xC+step,yC:yC+step] = 64;
                            imgB[xC:xC+step,yC:yC+step] = 64;
                    labCnt  += 1;
            
            img[:,:,0] = imgR;
            img[:,:,1] = imgG;
            img[:,:,2] = imgB;
            
            img = Image.fromarray(img);
            
            img = img.resize(disRes, Image.ANTIALIAS);            
            outImgName = os.path.join(ClassifyDir,alpha);
            img.save(outImgName);
            #cv2.imwrite(outImgName,img);            
            
            counter += 1;
            os.remove(alpha);

if __name__=='__main__':
    
    CharsDir    = "/home/kalyan/Python/PixelBased/bin/SyntheticData";
    ScaleSpace  = 15;
    TrainStepSz = 3;
    TestStepSz  = 1;
    
    TrainBinSz = 8;
    TestBinSz  = 8;
    
    #ReSizeSq          = (100,100);
    ReSizeSq         = (50,50);
    MarkResolution   = (2000,2000);
    PaintResolution   = (500,500);
    
    NUM_TRAIN_DATA   = 20;
    NUM_TEST_DATA    = 20;
    
    GENERATE       = 1;
    SIFT_DATA      = 0;
    DSIFT_DATA     = not(SIFT_DATA);
    BAYES_CLASSIFY = 1;

    separateTrainTest(CharsDir,NUM_TRAIN_DATA,NUM_TEST_DATA);
    
    FIRST_TIME = 1;
    for TrainBinSz in range(2,8,2):
        TestBinSz = TrainBinSz;
        getDenseSiftData('train', TrainBinSz, TrainStepSz,ReSizeSq);
        getDenseSiftData('test', TestBinSz, TestStepSz,ReSizeSq);
        
        feat, lab,xyCords                 = readFeaturesLabels('train',ReSizeSq,0,DSIFT_DATA);
        test_feat, test_lab, test_xyCords = readFeaturesLabels('test',ReSizeSq,0,DSIFT_DATA);
        classnames          = np.unique(lab);

        pca = PCA();
        
        V,S,m = pca.sparse(feat,20);
        
        features      = np.array([np.dot(V,f-m) for f in feat]);
        test_features = np.array([np.dot(V,f-m) for f in test_feat]);
        
        if(FIRST_TIME):
            Features      = features;
            Test_Features = test_features;
            
            Labels        = lab;
            Test_Labels   = test_lab;
            
            XY            = xyCords;
            Test_XY       = test_xyCords;
            
            Features_Append      = np.zeros(Features.shape);
            Test_Features_Append = np.zeros(Test_Features.shape);
            
            FIRST_TIME = 0;
        else:
            Features_AppendT      = np.copy(Features_Append);
            Test_Features_AppendT = np.copy(Test_Features_Append);
            
            nRows,_ = XY.shape;
            nCnt = 0;
            
            for i in range(nRows):
                if(nCnt < xyCords.shape[0]):
                    if(np.min(XY[i,:] == xyCords[nCnt,:])):
                        Features_AppendT[i,:] = features[nCnt,:];
                        nCnt +=1;
            
            nRows,_ = Test_XY.shape;
            nCnt = 0;
            
            for i in range(nRows):
                if(nCnt < test_xyCords.shape[0]):
                    if(np.min(Test_XY[i,:] == test_xyCords[nCnt,:])):
                        Test_Features_AppendT[i,:] = test_features[nCnt,:];
                        nCnt +=1;
            
            Features      = np.hstack((Features,Features_AppendT));
            Test_Features = np.hstack((Test_Features,Test_Features_AppendT));
        
    # test Bayes
    bc = BayesClassifier();
    blist = [Features[np.where(Labels == c)[0]] for c in classnames];
    bc.train(blist,classnames);
    res = bc.classify(Test_Features)[0];
    
    acc = sum(1.0*(res==Test_Labels)) / len(Test_Labels);
    print 'Accuracy:', acc
    
    printConfusion(res,Test_Labels,classnames,'Bayes');
#        markClassification(res,'test',ReSizeSq,MarkResolution);

    paintClassification(res,Test_Labels,'test',TestStepSz,ReSizeSq,PaintResolution,DSIFT_DATA);
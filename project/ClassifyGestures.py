# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 10:25:57 2015

@author: kalyan
"""

import os
import os.path
import shutil


import numpy as np

from lib import SIFT
from lib import KnnClassifier
from lib import PCA
from lib import BayesClassifier

def toNum(s):
    try:
        return int(s)
    except ValueError:
        return float(s)

def getGestureData(path,nFeat,sampStep,reSz):
    CharsDir  = "/home/kalyan/Python/ProgrammingComputerVision/data/gestures";
    CharsDir  = os.path.join(CharsDir, path);
    CharsList = os.listdir(CharsDir);
    CharsList.sort();
    
    counter = 1;

    cwd     = os.path.dirname(os.path.realpath(__file__));
    
    DataDir = os.path.join(cwd, '../bin/GestureData/');
    try:
        os.stat(DataDir);
    except:
        os.mkdir(DataDir);
    
    DataDir = os.path.join(DataDir, path);
    try:
        os.stat(DataDir);
    except:
        os.mkdir(DataDir);
    
    for char in CharsList:
        AlphaDir = os.path.join(CharsDir, char);
        AlphaList = os.listdir(AlphaDir);        
        AlphaList.sort();
        
        for alpha in AlphaList:                        
            wname = alpha.split('.');
            
            sift  = SIFT();
            fname = os.path.join(AlphaDir,alpha);
            cwd   = os.path.dirname(os.path.realpath(__file__));
            
            shutil.copy2(fname, cwd);
            
            SDES  = wname[0]+'.sift';
            
            sift.dense(alpha,SDES,nFeat,sampStep,resize=reSz);
            
    
            shutil.copy2(SDES, DataDir);
            
            os.remove(alpha);
            os.remove(SDES);

            counter += 1;
            print("================ %d. SIFT done for alphabet: %s ================\n" % (counter,alpha, ));

def readGestureFeaturesLabels(path):    

    cwd   = os.path.dirname(os.path.realpath(__file__));
    DataDir = os.path.join(cwd, '../bin/GestureData/');
    
    DataDir = os.path.join(DataDir, path);
    
    # create list of all files ending in .dsift
    featlist = [os.path.join(DataDir,f) for f in os.listdir(DataDir) if f.endswith('.sift')];
    
    # read the features
    features = [];
    sift  = SIFT();
    for featfile in featlist:
        loc,des = sift.readFeatures(featfile);
        features.append(des.flatten());
    
    features = np.array(features);
    
    # create labels
    labels = [featfile.split('/')[-1][0] for featfile in featlist];
    return features,np.array(labels);

def print_confusion(res,labels,classnames):
    
    n = len(classnames);

    # confusion matrix
    class_ind = dict([(classnames[i],i) for i in range(n)]);

    confuse = np.zeros((n,n));
    for i in range(len(labels)):
        confuse[class_ind[res[i]],class_ind[labels[i]]] += 1;

    print 'Confusion matrix for'
    print classnames;
    print confuse;

    
if __name__=='__main__':
    NoFeatures        = 10;
    DenseSampInterval = 5;
    ReSizeSq          = (50,50);
    
    DATA_GESTURE     = 0;
    KNN_CLASSIFY     = 0;
    BAYES_CLASSIFY   = 1;

    if(DATA_GESTURE):
        getGestureData('train',NoFeatures, DenseSampInterval, ReSizeSq);
        getGestureData('test',NoFeatures, DenseSampInterval, ReSizeSq);

    feat, lab           = readGestureFeaturesLabels('train');
    test_feat, test_lab = readGestureFeaturesLabels('test');
    classnames          = np.unique(lab);

    if(KNN_CLASSIFY):        
        k = 1;
        knn_classifier = KnnClassifier(lab,feat);
        res = np.array([knn_classifier.classify(test_feat[i],k) 
                    for i in range(len(test_lab))]);
        
        # accuracy
        acc = sum(1.0*(res==test_lab)) / len(test_lab);
        print 'Accuracy:', acc
        
        print_confusion(res,test_lab,classnames);
    
    if(BAYES_CLASSIFY):
        pca = PCA();
        V,S,m = pca.transform(feat);
        # keep most important dimensions
        V = V[:50];
        features      = np.array([np.dot(V,f-m) for f in feat]);
        test_features = np.array([np.dot(V,f-m) for f in test_feat]);
        
        # test Bayes
        bc = BayesClassifier();
        blist = [features[np.where(lab == c)[0]] for c in classnames];
        bc.train(blist,classnames);
        res = bc.classify(test_features)[0];
        
        acc = sum(1.0*(res==test_lab)) / len(test_lab);
        print 'Accuracy:', acc
        
        print_confusion(res,test_lab,classnames);
        
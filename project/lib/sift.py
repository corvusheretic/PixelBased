# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 08:21:35 2015

@author: kalyan
"""

import os
import os.path

import Image
import numpy as np

#import matplotlib.pylab as plt
from pylab import *

SIFTEXE_DIR  = "/home/kalyan/VlFeat/exe/bin/glnxa64";
SIFTEXE = os.path.join(SIFTEXE_DIR, 'sift');

class SIFT:

    def readFeatures(self,filename):
        """ Read feature properties and return in matrix form. """
        f = np.loadtxt(filename);
        return f[:,:4],f[:,4:]; # feature locations, descriptors
    
    def readExtFeatures(self,filename,isDense):
        """ Read feature properties and return in matrix form. """
        fSiftDes = open(filename, 'r');
        
        label = [];
        loc   = [];
        des   = [];
        
        if(isDense):
            for line in fSiftDes:
                lline = line.split(' ');
                if(len(lline) == 131):
                    label.append(lline[0]);
                    loc.append([lline[1],lline[2],'0','0']);
                    des.append(lline[5:-1]);
        else:
            for line in fSiftDes:
                lline = line.split(' ');
                if(len(lline) > 133):
                    label.append(lline[0]);
                    loc.append([lline[1],lline[2],lline[3],lline[4]]);
                    des.append(lline[5:-1]);
            
        return(label,loc,des);
    
    def writeFeatures(self,filename,locs,desc):
        """ Save feature location and descriptor to file. """
        np.savetxt(filename,(np.hstack((locs,desc))).astype(np.int));
    
    def plotFeatures(self,im,locs,circle=False):
        """ Show image with features. input: im (image as array),
        locs (row, col, scale, orientation of each feature). """
        def draw_circle(c,r):
            t = arange(0,1.01,.01)*2*pi
            x = r*cos(t) + c[0]
            y = r*sin(t) + c[1]
            plot(x,y,'b',linewidth=2);
        
        imshow(im)
        if circle:
            for p in locs:
                draw_circle(p[:2],p[2])
        else:
            plot(locs[:,0],locs[:,1],'ob');
        axis('off');
    
    def process(self,imagename,resultname,params="--edge-thresh 10 --peak-thresh 5"):
        """ Process an image and save the results in a file. """
        if imagename[-3:] != 'pgm':
            # create a pgm file
            im = Image.open(imagename).convert('L')
            im.save('tmp.pgm')
            imagename = 'tmp.pgm'
        cmmd = str(SIFTEXE+" "+imagename+" --output="+resultname+" "+params);
        os.system(cmmd);
        
        os.remove('tmp.pgm');
        print ('processed', imagename, 'to', resultname);
    
    def dense(self,imagename,resultname,size=20,steps=10,
              force_orientation=False,resize=None):
        """ Process an image with densely sampled SIFT descriptors
        and save the results in a file. Optional input: size of features,
        steps between locations, forcing computation of descriptor orientation
        (False means all are oriented upwards), tuple for resizing the image."""
        
        im = Image.open(imagename).convert('L');
        
        if resize!=None:
            im = im.resize(resize, Image.ANTIALIAS);
        
        m,n = im.size;
        if imagename[-3:] != 'pgm':
            # create a pgm file
            im.save('tmp.pgm');
            imagename = 'tmp.pgm';
    
        # create frames and save to temporary file
        scale = size/3.0;
        x,y = meshgrid(range(0,m,steps),range(0,n,steps));
        xx,yy = x.flatten(),y.flatten();
        frame = np.array([xx,yy,scale*ones(xx.shape[0]),zeros(xx.shape[0])]);
        np.savetxt('tmp.frame',frame.T,fmt='%03.3f');
    
        if force_orientation:
            cmmd = str(SIFTEXE+" "+imagename+" --output="+resultname+" --read-frames=tmp.frame --orientations");
        else:
            cmmd = str(SIFTEXE+" "+imagename+" --output="+resultname+" --read-frames=tmp.frame");
    
        os.system(cmmd);
        
        os.remove('tmp.pgm');
        os.remove('tmp.frame');
        print ('processed', imagename, 'to', resultname);
        
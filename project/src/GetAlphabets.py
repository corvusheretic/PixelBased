# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 14:12:55 2015

@author: kalyan
"""
import os
import os.path

import Image
import cv2
import numpy as np
import TextImg as ti

def GetbBox(ximg):
    
    indx = (ximg<255).nonzero();
    xmin,ymin = np.min(indx,1);
    xmax,ymax = np.max(indx,1);
    
    return (xmin,ymin,xmax,ymax);

if __name__=='__main__':
    FontsDir  = "/home/kalyan/Python/PixelBased/Fonts/Mixed";
    FontsList = os.listdir(FontsDir);
    FontsList.sort();
    
    A0         = 65;
    a0         = 97;
    nAplhabets = 26;
    FSize      = 15;#30;
    BImgSz     = 150;
    Magnify    = 4;
    BImgxPos   = 0.5*BImgSz;
    BImgyPos   = 0.25*BImgSz;
    border     = 5;
    
    for ascii in range(nAplhabets):
        counter=0;
        for FontName in FontsList:
            txt=chr(ascii+a0);
            TXT=chr(ascii+A0);
            
            cwd     = os.path.dirname(os.path.realpath(__file__));
            checkDir = os.path.join(cwd, 'SyntheticData/');
            try:
                os.stat(checkDir);
            except:
                os.mkdir(checkDir);
            
            checkDir1 = os.path.join(checkDir, txt);
            try:
                os.stat(checkDir1);
            except:
                os.mkdir(checkDir1);
            
            checkDir2 = os.path.join(checkDir, TXT);
            try:
                os.stat(checkDir2);
            except:
                os.mkdir(checkDir2);
            
            TIF_FILE1 = os.path.join(checkDir1,(txt+'_'+str(counter)+'.png'));
            TIF_FILE2 = os.path.join(checkDir2,(TXT+'_'+str(counter)+'.png'));

            fontName = os.path.join(FontsDir,FontName);

            if(counter==34):
                fontsize = 2*FSize;
            else:
                fontsize = FSize;
                
            pos      = (BImgxPos, BImgyPos);
            colorMap = (0,0,0);
            

            image = Image.new("RGBA", (BImgSz,BImgSz), (255,255,255));
            ti.CreateAlphabetImage(image, fontName, fontsize, txt, pos, colorMap);
            
            img_resized         = image.resize((Magnify*BImgSz,Magnify*BImgSz), Image.ANTIALIAS);
            np_img              = cv2.cvtColor(np.array(img_resized),cv2.COLOR_BGR2GRAY);
            ret,img             = cv2.threshold(np_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU);
            xmin,ymin,xmax,ymax = GetbBox(img);
            img = np_img[xmin-border:xmax+border,ymin-border:ymax+border];
            cv2.imwrite(TIF_FILE1, img);
            
            image = Image.new("RGBA", (BImgSz,BImgSz), (255,255,255));
            ti.CreateAlphabetImage(image, fontName, fontsize, TXT, pos, colorMap);            
            
            img_resized         = image.resize((Magnify*BImgSz,Magnify*BImgSz), Image.ANTIALIAS);
            np_img              = cv2.cvtColor(np.array(img_resized),cv2.COLOR_BGR2GRAY);
            ret,img             = cv2.threshold(np_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU);
            xmin,ymin,xmax,ymax = GetbBox(img);
            img = np_img[xmin-border:xmax+border,ymin-border:ymax+border];
            cv2.imwrite(TIF_FILE2, img);

            
            print("================ %d. Testing Font:%s for alphabets:(%s,%s) ================\n" % (counter,FontName,txt,TXT, ));
            counter +=1;
            print("Done\n");
            
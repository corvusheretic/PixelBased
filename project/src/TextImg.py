# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 14:12:55 2015

@author: kalyan
"""
import os
import os.path

import Image, ImageDraw, ImageFont
import cv2
import numpy as np

def CreateAlphabetImage(image, fontFile, fontSize, txt, txtPos, txtCMap):
    
    draw = ImageDraw.Draw(image);
    font = ImageFont.truetype(fontFile, fontSize);
    draw.text(txtPos, txt, txtCMap, font=font);
    
if __name__=='__main__':
    FontsDir  = "/home/kalyan/Python/PixelBased/Fonts/Handwriting";    
    FontsList = os.listdir(FontsDir);
    FontsList.sort();
    
    for FontName in FontsList:
        txt='the quick brown fox jumped over the lazy dog.';
        TXT='THE QUICK BROWN FOX JUMPED OVER THE LAZY DOG.';
        TIF_FILE='test.png';
        
        image = Image.new("RGBA", (2000,1000), (255,255,255));
        fontName = os.path.join(FontsDir,FontName);
        fontsize=50;
        pos1 = (100, 0);
        pos2 = (100, 100);
        colorMap = (0,0,0);
        
        CreateAlphabetImage(image, fontName, fontsize, txt, pos1, colorMap);
        CreateAlphabetImage(image, fontName, fontsize, TXT, pos2, colorMap);
        
        cv2.imwrite(TIF_FILE, np.asarray(image));
        print("================ Testing Font:%s ================\n" % (FontName, ));
        print("Done\n");
        
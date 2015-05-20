# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 11:32:02 2015

@author: kalyan
"""

import os
import os.path
import shutil

import numpy as np
from PIL import Image
import cv2

def classHeatmap(alpha,imgSZ,scale,xpdf,magnify):
    
    (nImgRow,nImgCol) = (4,14);
    cwd   = os.path.dirname(os.path.realpath(__file__));
    SaveDir = os.path.join(cwd, '../nbin/AlphabetData/heatmap');    

    wname = alpha.split('.');
    ffile = os.path.join(SaveDir, wname[0]+'_'+str(scale)+'.'+wname[1]);
    pimg = cv2.cvtColor(cv2.imread(ffile),cv2.COLOR_RGB2GRAY);
    pimgSZ = pimg.shape;
    
    canvasSZ  = (nImgRow*magnify*imgSZ[1],nImgCol*magnify*imgSZ[0]);
    tcanvasSZ = (nImgRow*imgSZ[1],nImgCol*imgSZ[0]);
    
    CONTENT="""<!DOCTYPE html>
<!--
To change this license header, choose License Headers in Project Properties.
To change this template file, choose Tools | Templates
and open the template in the editor.
-->
<html>
    <head>
        <title>""";
    
    CONTENT +=wname[0]+'_'+str(scale)+""" Color map</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <script type="text/javascript" src="../../dist/jquery.js"></script>
        <script type="text/javascript" src="../../dist/plugins/rainbow/rainbowvis.js"></script>
		<link rel="stylesheet" type="text/css" href="../../dist/jquery.jqplot.css" />
		<script language="javascript" type="text/javascript" src="../../dist/jquery.jqplot.js"></script>
		<script language="javascript" type="text/javascript" src="../../dist/plugins/jqplot.canvasTextRenderer.min.js"></script>
		<script language="javascript" type="text/javascript" src="../../dist/plugins/jqplot.canvasAxisLabelRenderer.min.js"></script>
		<script language="javascript" type="text/javascript" src="../../dist/plugins/jqplot.dateAxisRenderer.js"></script>
		<script language="javascript" type="text/javascript" src="../../dist/plugins/jqplot.categoryAxisRenderer.js"></script>
		<script language="javascript" type="text/javascript" src="../../dist/plugins/jqplot.highlighter.js"></script>
		<script language="javascript" type="text/javascript" src="../../dist/plugins/jqplot.cursor.js"></script>
        <style type="text/css">
            table { border-collapse:collapse; }
            table,th, td { border: 1px solid black; }
        </style>
    </head>
    <body>
		<div style="position: relative;">
			<div id="hLegend" style="left: 50; top: 10;" ><h2>Legend</h2></div>
			<canvas id="Legend" width="400" height="30" style="left: 50; top: 30; border:1px solid #c3c3c3;"></canvas>
			<div id="hHeatmap" style="left: 50; top: 90;" ><h2>""";
    CONTENT +=wname[0]+'_scale'+str(scale)+"""</h2></div>
			<canvas id="Background" width=\""""+str(canvasSZ[0])+'\" height=\"'+str(canvasSZ[1])+"""\" style="left: 0; top: 120; border:1px solid #c3c3c3;"></canvas>
			<canvas id="Temp" width=\""""+str(tcanvasSZ[0])+'\" height=\"'+str(tcanvasSZ[1])+'\" style="left: 0; top: '+str(canvasSZ[1]+130)+""";"></canvas>
			<div id="plot" style="left: """+str(canvasSZ[0]+20)+"""; border:1px dashed #ff0000;" ></div>
		</div>
        <script type="text/javascript">
        //////////////////////////////////////////////////////////////////
        """;
    
    pdf = np.zeros((53,imgSZ[0],imgSZ[1]));
    for i in range(xpdf.shape[1]):
        #col = -np.log(-xpdf[:,i] +1);
        col = xpdf[:,i];
        col = col.reshape(imgSZ,order='F');
        
        pdf[i] = col;
    
    wpdf    = np.min(pdf)*np.ones(imgSZ);
    bkg_pdf = pdf[0];
    bkg_pdf = np.hstack((bkg_pdf,wpdf));
    
    wpdf    = np.hstack((wpdf,wpdf));                
    img_pdf = np.hstack((bkg_pdf,wpdf));
    cnt = 1;
    
    for i in range(13):
        row_pdf = pdf[cnt];
        cnt +=1;
        for j in range(3):
            row_pdf = np.hstack((row_pdf,pdf[cnt]));
            cnt +=1;
        
        img_pdf = np.vstack((img_pdf,row_pdf));
    
    np.savetxt('test.txt', img_pdf, fmt='%d', delimiter=',',newline=',');
    
    with open ('test.txt','r') as myfile:
        data = myfile.read();
    myfile.close();
    
    strs = data.split(',');
    strs[0] = '             var pdf'+' = ['+strs[0];
    strs[-1] = strs[-1]+'];\n';
    strs = ','.join(strs);             
    CONTENT += strs;

    imgSZ     = list(imgSZ);
    pimgSZ    = list(pimgSZ);
    canvasSZ  = list(canvasSZ); 
    tcanvasSZ = list(tcanvasSZ);
    
    CONTENT +="""

			classStr = [['Bkg','','',''],
			['A','B','C','D'],
			['E','F','G','H'],
			['I','J','K','L'],
			['M','N','O','P'],
			['Q','R','S','T'],
			['U','V','W','X'],
			['Y','Z','a','b'],
			['c','d','e','f'],
			['g','h','i','j'],
			['k','l','m','n'],
			['o','p','q','r'],
			['s','t','u','v'],
			['w','x','y','z']]; 
            
            var imgSZ = """ + str(imgSZ)+';\n';
    CONTENT += '            var pimgSZ = ' + str(pimgSZ)+';\n';
    CONTENT += '            var canvasSZ = ' + str(canvasSZ)+';\n';
    CONTENT += '            var tcanvasSZ = ' + str(tcanvasSZ)+';\n';
    CONTENT += '            var magnify = '+str(magnify)+';\n';
			
    CONTENT += """
            $(document).ready(function(){
                
                function hexToRgb(hex) {
                    var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
                    return result ? {
						r: parseInt(result[1], 16),
						g: parseInt(result[2], 16),
						b: parseInt(result[3], 16)
                    } : null;
                };
				
				    $('#Background').click(function(e) {
                    
                            //////////////////////////// Handle Scale '+str(sloop)+' ////////////////////////////
                            var pdfmax = Math.max.apply(null, pdf);
							var pdfmin = Math.min.apply(null, pdf);

							var heatmap = new Rainbow(); 
							heatmap.setSpectrum('white', 'blue', 'green','yellow', 'red');
							heatmap.setNumberRange(pdfmin, pdfmax);
							
							var canvas = document.getElementById('Temp');
							var ctx    = canvas.getContext('2d');
							var img    = new Image();
							img.src = '""";
    CONTENT +=wname[0]+'_bkg.'+wname[1]+'\';\n';
    CONTENT +="""							ctx.drawImage(img, 0, 0,canvasSZ[0],canvasSZ[1],0,0,pimgSZ[0],pimgSZ[1]);

							var imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

							var pixels = imageData.data;
							
							for( var i = 0; i < pdf.length; i++) {
								var hexColour = '#' + heatmap.colourAt(pdf[i]);
								var cc = hexToRgb(hexColour);
								pixels[i*4]   = cc.r; // Red
								pixels[i*4+1] = cc.g; // Green
								pixels[i*4+2] = cc.b; // Blue
							};
							
							ctx.clearRect(0, 0, canvas.width, canvas.height);
							ctx.putImageData(imageData, 0, 0);
							
							var canvas1 = document.getElementById('Background');
							var ctx1    = canvas1.getContext('2d');
							ctx1.clearRect(0, 0, canvas1.width, canvas1.height);
							ctx1.drawImage(img, 0, 0);
							ctx1.globalAlpha = 0.6;
							
							for (var x = 0; x < canvas1.width; x+=magnify*imgSZ[1]) {
								for (var y = 0; y < canvas1.height; y+=magnify*imgSZ[0]) {
									var str = classStr[Math.floor(y/(magnify*imgSZ[0]))][Math.floor(x/(magnify*imgSZ[1]))];
									ctx1.fillStyle = '#000';
									ctx1.font = 'bold 20px verdana';
									ctx1.fillText(str, x + 20, y + 20, 60);
								}
							}
							
							ctx1.drawImage(canvas, 0, 0, canvas.width*4.0, canvas.height*4.0);
							ctx.clearRect(0, 0, canvas.width, canvas.height);
							
							var img2   = new Image();
							img2.src = '""";
           
    CONTENT +=wname[0]+'_'+str(scale)+'.'+wname[1]+'\';\n';
    CONTENT +="""							ctx1.drawImage(img2, 0, 0,pimgSZ[1],pimgSZ[0],3*magnify*imgSZ[1],0,magnify*imgSZ[1],magnify*imgSZ[0]);
							
							var canvasOffset = $(canvas1).offset();
							var canvasX = Math.floor(e.pageX-canvasOffset.left);
							var canvasY = Math.floor(e.pageY-canvasOffset.top);
							
							// Get relative x,y cordinates in each square
							var relX = canvasX - Math.floor(canvasX/(magnify*imgSZ[1]))*(magnify*imgSZ[1]);
							var relY = canvasY - Math.floor(canvasY/(magnify*imgSZ[0]))*(magnify*imgSZ[0]);
							
							var imageData = ctx1.getImageData(0, 0, canvas1.width, canvas1.height);
							var pixels = imageData.data;
							
							var indexX = Math.floor(relX/magnify);
							var indexY = Math.floor(relY/magnify);

							//alert("X: " + canvasX + ", Y: " + canvasY + " <====> Xb: " + relX + ", Yb: " + relY + " <====> Xi: " + indexX + ", Yi: " + indexY);
							
							// First handle all but the first row
							for (var sqX = 0; sqX < canvas1.width; sqX +=magnify*imgSZ[1]) {
								for (var sqY = 200; sqY < canvas1.height; sqY +=magnify*imgSZ[0]) {
									for (var x = sqX + indexX*magnify; x < sqX + (indexX+1)*magnify; x++) {
										for (var y = sqY + indexY*magnify; y < sqY + (indexY+1)*magnify; y++) {
											var pixelRedIndex = ((y - 1) * (imageData.width * 4)) + ((x - 1) * 4);
											pixels[pixelRedIndex] = 0; // Red
											pixels[pixelRedIndex+1] = 0; // Green
											pixels[pixelRedIndex+2] = 0; // Blue
										};
									};
								};
							};
							// Handle the first row
							for (var sqX = 0; sqX < canvas1.width; sqX +=3*magnify*imgSZ[1]) {
								for (var sqY = 0; sqY < canvas1.height; sqY +=magnify*imgSZ[0]) {
									for (var x = sqX + indexX*magnify; x < sqX + (indexX+1)*magnify; x++) {
										for (var y = sqY + indexY*magnify; y < sqY + (indexY+1)*magnify; y++) {
											var pixelRedIndex = ((y - 1) * (imageData.width * 4)) + ((x - 1) * 4);
											pixels[pixelRedIndex] = 0; // Red
											pixels[pixelRedIndex+1] = 0; // Green
											pixels[pixelRedIndex+2] = 0; // Blue
										};
									};
								};
							};
							ctx1.putImageData(imageData, 0, 0);
							
							var classPDF = [];
							classPDF.push(pdf[indexY*imgSZ[1]*"""+str(nImgRow)+""" + indexX]);
							for (var indexOffsetY = 1; indexOffsetY < """+str(nImgCol)+"""; indexOffsetY++) {
								for (var indexOffsetX = 0; indexOffsetX < """+str(nImgRow)+"""; indexOffsetX++) {
									classPDF.push(pdf[ indexOffsetY*imgSZ[0]*imgSZ[1]*"""+str(nImgRow)+""" +indexY*imgSZ[1]*"""+str(nImgRow)+""" + indexOffsetX*imgSZ[1] + indexX]);  
								}
							}
							
                                       //alert(classPDF);
                                       
							/////////////////////////////
							// Set the pdf graph index //
							/////////////////////////////
                            var plot = $.jqplot('plot', [classPDF], {
								// The "seriesDefaults" option is an options object that will
								// be applied to all series in the chart.
								seriesDefaults: {
									rendererOptions: {smooth: true}
								},
								// Custom labels for the series are specified with the "label"
								// option on the series option.  Here a series option object
								// is specified for each series.
								series: [
									{label: 'Prob. Dist. Function'}
								],
								// Show the legend and put it outside the grid, but inside the
								// plot container, shrinking the grid to accomodate the legend.
								// A value of "outside" would not shrink the grid and allow
								// the legend to overflow the container.
								legend: {
												show: true,
												placement: 'outsideGrid'
											},
								axes: {
									// Use a category axis on the x axis and use our custom ticks.
									xaxis: {
										ticks: [[1,"BkG"], [2,"A"], [3,"B"], [4,"C"], [5,"D"], [6,"E"], [7,"F"], [8,"G"],
											[9,"H"], [10,"I"], [11,"J"], [12,"K"], [13,"L"], [14,"M"], [15,"N"], [16,"O"],
											[17,"P"], [18,"Q"], [19,"R"], [20,"S"], [21,"T"], [22,"U"], [23,"V"], [24,"W"],
											[25,"X"], [26,"Y"], [27,"Z"], [28,"a"], [29,"b"], [30,"c"], [31,"d"], [32,"e"],
											[33,"f"], [34,"g"], [35,"h"], [36,"i"], [37,"j"], [38,"k"], [39,"l"], [40,"m"],
											[41,"n"], [42,"o"], [43,"p"], [44,"q"], [45,"r"], [46,"s"], [47,"t"], [48,"u"],
											[49,"v"], [50,"w"], [51,"x"], [52,"y"], [53,"z"]],
											tickInterval: 1
									},
									// Pad the y axis just a little so bars can get close to, but
									// not touch, the grid boundaries.  1.2 is the default padding.
									yaxis: {
										pad: 1.05,
										tickOptions: {formatString: '%1.2g',showGridline: false}
									}
								}
							});
							plot.replot();
						});
				try {
						var pdfmax = Math.max.apply(null, pdf);
						var pdfmin = Math.min.apply(null, pdf);

						var heatmap = new Rainbow(); 
						heatmap.setSpectrum('white','blue', 'aqua', 'green' ,'lime', 'yellow', 'red');
						heatmap.setNumberRange(pdfmin, pdfmax);
						
						var canvas = document.getElementById('Temp');
						var ctx    = canvas.getContext('2d');
						var img    = new Image();
						img.src = '""";
    CONTENT += wname[0]+'_bkg.'+wname[1]+'\';\n';
    CONTENT += """						ctx.drawImage(img, 0, 0,canvasSZ[0],canvasSZ[1],0,0,tcanvasSZ[0],tcanvasSZ[1]);

						var imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

						var pixels = imageData.data;
						
						for( var i = 0; i < pdf.length; i++) {
							var hexColour = '#' + heatmap.colourAt(pdf[i]);
							var cc = hexToRgb(hexColour);
							pixels[i*4]   = cc.r; // Red
							pixels[i*4+1] = cc.g; // Green
							pixels[i*4+2] = cc.b; // Blue
						};
						
						ctx.clearRect(0, 0, canvas.width, canvas.height);
						ctx.putImageData(imageData, 0, 0);
						
						var canvas1 = document.getElementById('Background');
						var ctx1    = canvas1.getContext('2d');
						ctx1.drawImage(img, 0, 0);
						ctx1.globalAlpha = 0.6;
						
						for (var x = 0; x < canvas1.width; x+=magnify*imgSZ[1]) {
							for (var y = 0; y < canvas1.height; y+=magnify*imgSZ[0]) {
								var str = classStr[Math.floor(y/(magnify*imgSZ[0]))][Math.floor(x/(magnify*imgSZ[1]))];
								ctx1.fillStyle = '#000';
								ctx1.font = 'bold 20px verdana';
								ctx1.fillText(str, x + 20, y + 20, 60);
							}
						}
						
						ctx1.drawImage(canvas, 0, 0, canvas.width*4.0, canvas.height*4.0);
						ctx.clearRect(0, 0, canvas.width, canvas.height);
						
                                 	var img2   = new Image();
						img2.src = '""";
           
    CONTENT +=wname[0]+'_'+str(scale)+'.'+wname[1]+'\';\n';
    CONTENT +="""							ctx1.drawImage(img2, 0, 0,pimgSZ[1],pimgSZ[0],3*magnify*imgSZ[1],0,magnify*imgSZ[1],magnify*imgSZ[0]);
						
						var canvasLegend = document.getElementById('Legend');
						var contextLegend = canvasLegend.getContext('2d');
						
						var rainbowLegend = new Rainbow();
						rainbowLegend.setSpectrum('white', 'blue', 'green', 'yellow', 'red');
						rainbowLegend.setNumberRange(0, canvasLegend.width);
						
						contextLegend.lineWidth = 1;
						for (var i = 0; i <= canvasLegend.width; i++) {
							contextLegend.beginPath();
							contextLegend.moveTo(i,0);
							contextLegend.lineTo(i,canvasLegend.height);
							contextLegend.strokeStyle = '#' + rainbowLegend.colourAt(i); 
							contextLegend.stroke();
						}
						contextLegend.font = 'bold 20px verdana';
						contextLegend.fillText(pdfmin, 0, canvasLegend.height, 60);
						contextLegend.fillText(pdfmax, canvasLegend.width-20, canvasLegend.height, 60);
						
					} catch (err) {
						alert(err);
					}
			});
        </script>
    </body>
</html>""";
        
    html = os.path.join(SaveDir,wname[0]+'_'+str(scale)+'.html');
    myfile = open (html,'w');
    myfile.write(CONTENT);
    myfile.close();
    
    return(pdf);


#np.finfo(np.double).tiny 
def heatmapBkgLayer(alpha,sz,magnify):
    cwd   = os.path.dirname(os.path.realpath(__file__));
    SaveDir = os.path.join(cwd, '../nbin/AlphabetData/heatmap');
    
    ffile = os.path.join(cwd, alpha);
    
    img = Image.open(ffile).convert('L');
    img = img.resize((magnify*sz[1],magnify*sz[0]), Image.ANTIALIAS);
    img = np.array(img);
    
    img = 255*((img == 255).astype(np.uint8)).astype(np.uint8);
    
    wimg = (255*np.ones(img.shape)).astype(np.uint8);
    
    imgRow = np.hstack((img,img));
    imgRow = np.hstack((imgRow,imgRow));
    
    imgCanvas = imgRow;
    for i in range(12):
        imgCanvas = np.vstack((imgCanvas,imgRow));
    
    bkg_row = np.hstack((wimg,wimg));
    bkg_row = np.hstack((bkg_row,wimg));
    bkg_row = np.hstack((img,bkg_row));
    
    imgCanvas = np.vstack((bkg_row,imgCanvas));
    
    wname = alpha.split('.');
    fname = os.path.join(SaveDir, wname[0]+'_bkg'+'.'+wname[1]);
    
    cv2.imwrite(fname,imgCanvas);
        
if __name__=='__main__':    
    
    START_BINSZ = 4;
    END_BINSZ = 6;
    STEP_BINSZ = 2;
    
    if(0):
        pointPDF(START_BINSZ,END_BINSZ,STEP_BINSZ,10);
    
    if(0):
        cwd   = os.path.dirname(os.path.realpath(__file__));
        SaveDir = os.path.join(cwd, '../bin/AlphabetData/heatmap');
        try:
            os.stat(SaveDir);
        except:
            os.mkdir(SaveDir);
        
        shutil.rmtree(SaveDir);
        os.mkdir(SaveDir);
        heatmapBkgLayer((200,200));
        print('Bkg Layers ready.\n');
    
    classHeatmap(START_BINSZ,END_BINSZ,STEP_BINSZ);
    
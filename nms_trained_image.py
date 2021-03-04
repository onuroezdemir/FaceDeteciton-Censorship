# -*- coding: utf-8 -*-

import numpy as np
import cv2

# "imread" function is reads the image
image = cv2.imread('YOUR_IMAGE_PATH')
#print(image)

#imageH -> height of image *** imageW -> width of image ------- shape function is output of pixels and rgb size
imageH = image.shape[0]
imageW = image.shape[1]

# blobFromImage -> converts the image to 4D tensors format(blob)*** 1/255 is Yolo's advice, (416,416) is used format, swapRB is BGR to RGB
image_blob = cv2.dnn.blobFromImage(image, 1/255, (416,416), swapRB=True,crop=False)

label = ["face_here"]

#cfg and weeights files to model
model = cv2.dnn.readNetFromDarknet("YOUR_PATH/trained_model/face_yolov4.cfg",
                                   "YOUR_PATH/face_yolov4_last.weights")

#outputs layers -- conv values
layers = model.getLayerNames()
outputLayer = [ layers[layer[0]-1] for layer in model.getUnconnectedOutLayers()]

model.setInput(image_blob)

detectionLayers = model.forward(outputLayer)

#NMS
idsList =[]
boxesList=[]
confidenceList=[]


for detectionLayer in detectionLayers:
    for objectDetection in detectionLayer:
        
        scores = objectDetection[5:]
        predictedId = np.argmax(scores)
        
        confidenceScore = scores[predictedId]
        
        if confidenceScore > 0.25:
            
            lbl = label[predictedId]
            
            boundingBox = objectDetection[0:4]*np.array([imageW,imageH,imageW,imageH])
            
            (boxCenterX, boxCenterY, boxW, boxH) = boundingBox.astype("int")
            
            startX = int(boxCenterX- (boxW/2))
            startY = int(boxCenterY-(boxH/2))
            
            
############NMS###############
            
            idsList.append(predictedId)
            confidenceList.append(float(confidenceScore))
            boxesList.append([startX,startY, int(boxW), int(boxH)])
                        
########NMS ##################
                       
maxIds = cv2.dnn.NMSBoxes(boxesList,confidenceList,0.5,0.4)
         
for maxId in maxIds:
    maxClassId = maxId[0]
    box = boxesList[maxClassId]
    
    startX = box[0]
    startY = box[1]
    boxW = box[2]
    boxH = box[3]
        
    predictedId = idsList[maxClassId]
    lbl = label[predictedId]
    confidenceScore = confidenceList[maxClassId]
      
    endX = startX + boxW
    endY = startY +boxH
                                                  
    roi =  image[startY:endY,startX:endX]
    blur = cv2.GaussianBlur(roi,(65,65),0)
    image[startY:endY,startX:endX]=blur                        
                                     
    label = "{}: {:.2f}%".format(label, confidenceScore*100)
    print("predicted object {}".format(label))                        

cv2.imshow('Detected',image)

























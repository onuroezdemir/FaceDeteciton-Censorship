# -*- coding: utf-8 -*-

import numpy as np
import cv2

cap = cv2.VideoCapture("YOUR_VIDEO_PATH")

while True:
    ret, frame = cap.read()
    
    frame = cv2.flip(frame,1)
    frame= cv2.resize(frame, (960,720))
    
    frameH= frame.shape[0]
    frameW=frame.shape[1]

    frame_blob = cv2.dnn.blobFromImage(frame, 1/255, (416,416), swapRB=True,crop=False)

    label = ["face_here"]

#cfg and weeights files to model
    model = cv2.dnn.readNetFromDarknet("YOUR_PATH/trained_model/face_yolov4.cfg",
                                   "YOUR_PATH/trained_model/face_yolov4_last.weights")

#outputs layers -- conv values
    layers = model.getLayerNames()
    outputLayer = [ layers[layer[0]-1] for layer in model.getUnconnectedOutLayers()]
    
    model.setInput(frame_blob)
    
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
                
                boundingBox = objectDetection[0:4]*np.array([frameW,frameH,frameW,frameH])
                
                (boxCenterX, boxCenterY, boxW, boxH) = boundingBox.astype("int")
                
                startX = int(boxCenterX- (boxW/2))
                startY = int(boxCenterY-(boxH/2))
                                
 ############NMS###############
                
                idsList.append(predictedId)
                confidenceList.append(float(confidenceScore))
                boxesList.append([startX,startY, int(boxW), int(boxH)])
                                       
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
                  
        roi =  frame[startY:endY,startX:endX]
        blur = cv2.GaussianBlur(roi,(65,65),0)
        frame[startY:endY,startX:endX]=blur
                
        label = "{}: {:.2f}%".format(label, confidenceScore*100)
        print("predicted object {}".format(label))
                     
    cv2.imshow('Detected',frame)

    if cv2.waitKey(1) & 0xFF ==ord("q"):
        break

cap.release()
cv2.destroAllWindows()


























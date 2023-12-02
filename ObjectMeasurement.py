import cv2
import numpy as np
import utils

webcam = True
path = 'D:/Projects/Measurements/img.png'
cap = cv2.VideoCapture(1)
cap.set(10,160)
cap.set(3,1920)
cap.set(4,1080)
scale = 2
wP = 297*scale
hP = 210*scale

while True:
    if webcam: success,img = cap.read()
    else: img = cv2.imread(path)

    imgContours, conts = utils.getContours(img, minArea=50000, filter=4, draw=True)
    if len(conts)!= 0:
        biggest = conts[0][2]
        #print(biggest)
        imgWarp = utils.warpImage(img, biggest, wP, hP)
        imgContours2, conts2 = utils.getContours(imgWarp, minArea=2000, filter=4, cThr=[50,50], draw=False)
        if len(conts2)!=0:
            for obj in conts2:
                cv2.polylines(imgContours2,[obj[2]],True,(0,255,0),2)

                nPoints = utils.reorder(obj[2])
                nw = round((utils.findDis(nPoints[0][0]//scale, nPoints[1][0]//scale)/10),2)
                nH = round((utils.findDis(nPoints[0][0]//scale, nPoints[2][0]//scale)/10),2)

                # Draw arrowed line and put text on the contour lines
                cv2.arrowedLine(imgContours2, tuple(nPoints[0][0]), tuple(nPoints[1][0]), (255, 0, 255), 2)
                cv2.arrowedLine(imgContours2, tuple(nPoints[0][0]), tuple(nPoints[2][0]), (255, 0, 255), 2)
                cv2.putText(imgContours2, f'{nw}cm', tuple(nPoints[0][0]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255),1)
                cv2.putText(imgContours2, f'{nH}cm', tuple(nPoints[2][0]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255),1)

        cv2.imshow('A4', imgContours2)

        #cv2.imshow('A4', imgWarp)

    img = cv2.resize(img,(0,0),None,0.6,0.6)

    cv2.imshow('Original',img)
    cv2.waitKey(1)
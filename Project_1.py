import cv2
import numpy as np
import Project_2

#Read the image
Image = "D:\\New folder\\OMR\\testCase\\test3.jpg"
ImgWidth = 700
Imgheight = 700
questions = 5
options = 4
solu = [0, 1, 2, 2, 3]

img = cv2.imread(Image)
img = cv2.resize(img, (ImgWidth, Imgheight))#Resize Image
imgBigCont = img.copy()
imgCont = img.copy()
finalImg = img.copy()

#Perform Preprocessing in image
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#Gray Image
blurImg = cv2.GaussianBlur(grayImg, (5, 5), 1)#Blur Gray Image
cannyImg = cv2.Canny(blurImg, 10, 50)#Canny Image

cont, hierarchy = cv2.findContours(
    cannyImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)# Find all Contours
rectCon = Project_2.rectCont(cont) #Filter rectangles from Countours
bigCont = Project_2.getCornerPoints(rectCon[0])#Corner points of biggest rectangle(For OMR box)
gradeScore = Project_2.getCornerPoints(rectCon[1])#Corner points of secnod biggest rectangle(For grade box)
bigCont = Project_2.reorder(bigCont)
gradeScore = Project_2.reorder(gradeScore)

pt1 = np.float32(bigCont)
pt2 = np.float32(
    [[0, 0], [ImgWidth, 0], [0, Imgheight], [ImgWidth, Imgheight]])
matrix = cv2.getPerspectiveTransform(pt1, pt2)
imgCol = cv2.warpPerspective(img, matrix, (ImgWidth, Imgheight))

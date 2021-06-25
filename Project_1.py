import cv2
import numpy as np

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


import cv2
import numpy as np


def rectCont(cont):#Check if given contour is rectangle
    rectCon = []
    for i in cont:
        area = cv2.contourArea(i)
        if area > 50:
            perimeter = cv2.arcLength(i, True)
            sides = cv2.approxPolyDP(i, 0.02 * perimeter, True)
            if len(sides) == 4:
                rectCon.append(i)
    rectCon = sorted(rectCon, key=cv2.contourArea, reverse=True)
    return rectCon


def getCornerPoints(cont):#Find the corners of given the rectangle
    perimeter = cv2.arcLength(cont, True)
    sides = cv2.approxPolyDP(cont, 0.02 * perimeter, True)
    return sides


def reorder(points):#Arrange all the points of rectangle 
    points = points.reshape((4, 2))
    newPoints = np.zeros((4, 1, 2), np.int32)
    add = points.sum(1)
    newPoints[0] = points[np.argmin(add)]
    newPoints[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    newPoints[1] = points[np.argmin(diff)]
    newPoints[2] = points[np.argmax(diff)]
    return newPoints

# Function to split the image in boxes 
def splitBoxes(img, questions, options):
    rows = np.vsplit(img, questions)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, options)
        for box in cols:
            boxes.append(box)
    return boxes

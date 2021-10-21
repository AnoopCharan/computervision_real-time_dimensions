# from typing import final
import numpy as np
import cv2 as cv

# function to get required contours from chosen frame 
def get_contours(img, cThr=(100,175), showCanny =False, minArea =1000, filter= 0, draw= False):
    # convert image to grey scale 
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Apply a blur 
    blur = cv.GaussianBlur(gray,(5,5), 1)
    # Apply canny edge detector
    canny = cv.Canny(blur, cThr[0], cThr[1])
    # add dialation
    kernal = np.ones((5,5))
    dialate = cv.dilate(canny,kernal, iterations=3)
    # erode
    erode = cv.erode(dialate, kernal, iterations=2 )
    if showCanny : cv.imshow('Canny', erode)
    # find contours 
    contours, hierarchy = cv.findContours(erode, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #  empty list to collect required contours 
    final_contours= []
    # iterate over each contour to check and get required contours 
    for i in contours:
        # calculate area of current contour (contours are closed curves)
        area =cv.contourArea(i)
        # check if the area is passing above the defined limit
        if area > minArea:
            # calculate the perimeter/arc length of the contour
            perimeter = cv.arcLength(i, closed=True)
            #  appromiate the contour, lower epsilon fits the contour closely: 2% of perimeter is chosen
            approx = cv. approxPolyDP(i, 0.02*perimeter, closed= True )
            # get a bounding box array for the contour
            bounbbox = cv.boundingRect(approx)
            #  determines if the contours match required spec, filter is no of corners (rectangles, circles etc)
            # append passing contour parameters to final contours list
            if filter > 0:
                if len(approx) == filter:
                    final_contours.append([len(approx), area, approx, bounbbox, i])
            else:
                final_contours.append([len(approx), area, approx, bounbbox, i])
    
    # Sort the contours based on length of array in decending order, first contour will be larges and should be the guide
    # key can take only functions, so lambda is used to get len(approx) as key
    final_contours =sorted(final_contours, key= lambda x:x[1], reverse=True)
    # display contours based on request
    if draw:
        for con in final_contours:
            cv.drawContours(img, con[4],-1, (0,0,255), 3 )

    # finally returns the img, and list of final contours
    return img, final_contours

# function to reorder the corner points of contour
def reorder (mypoints):
    # make a zeros array of same shape as input array
    mypoints_new = np.zeros_like(mypoints)
    # change shape of input array to 2 level array
    mypoints = mypoints.reshape((4,2))
    # get array with sum of each internal list 
    add= mypoints.sum(1)
    # get the points with min value as [0] for least, max as [3] for opposite (h + w)
    mypoints_new[0] = mypoints[np.argmin(add)]
    mypoints_new[3] = mypoints[np.argmax(add)]
    # get sequential difference of mypoints in axis =1 ex: [1,2,3,4] --> [1,1,1,-4] a[i+1]-a[i]
    dif = np.diff(mypoints, axis=1)
    #  get points with least and max difference for points 1 and 2 
    mypoints_new[1] = mypoints[np.argmin(dif)]
    mypoints_new[2] = mypoints[np.argmax(dif)]
    return mypoints_new


#  function to warp image and make it flat
def warpimg(img, points, w, h, pad=20):
    # print(points)
    # use reorder function to reorder points of contours
    points = reorder(points)
    # print(points)
    # set reordered points as float 32
    pts1 = np.float32(points)
    # set the new shape to transform the image to (size of guide)
    pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    # get matrix for using in warping
    matrix = cv.getPerspectiveTransform(pts1, pts2)
    # use matrix to warp perspective 
    imgwarp = cv.warpPerspective(img, matrix, (w,h))
    # use padding to remove any non guide edges
    imgwarp = imgwarp[ pad: imgwarp.shape[0]-pad, pad: imgwarp.shape[1]-pad]

    return imgwarp

# function to calulate mangnitudes
def findist(pts1, pts2):
    #  SQRT((x2-x1)^2 + (y2-y1)^2)
    return ((pts2[0]-pts1[0])**2 + (pts2[1]-pts1[1])**2)**0.5




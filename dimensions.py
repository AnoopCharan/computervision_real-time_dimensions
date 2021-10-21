import cv2 as cv 
import numpy as np
import utilis


webcam = False

capture = cv.VideoCapture(0+cv.CAP_DSHOW)
scale = 3
guide_width =210*scale
guide_height= 297*scale





while True:
    if webcam == True:
        isTrue, img = capture.read()
    else:
        img = cv.imread('3.jpg')
    img = cv.resize(img,(0,0), None,0.25,0.25) 
    imgcont, conts = utilis.get_contours(img, minArea=50000,cThr=(100,150), filter=4, draw=True)
    print(len(conts))

    # cv.imshow('test', imgcont)
    # cv.waitKey(0)

    if len(conts) != 0:
        bigcont = conts[0][2]
        # print(bigcont)
    imgwarp = utilis.warpimg(img, bigcont, guide_width, guide_height)
    # cv.imshow('warped image', imgwarp)

    imgcont2, conts2 = utilis.get_contours(imgwarp, minArea=1000, filter=4,cThr=(50,50),draw=False)
    print('cont2', len(conts2))
    if len(conts2) !=0:
        for obj in conts2:
            # draw lines using the end points 
            cv.polylines(imgcont2, [obj[2]], True, (0,255,0), 3 )
            # reorder corner coordinates list 0-3        
            newpoints = utilis.reorder(obj[2])
            # print(newpoints.shape)
            nW= round(utilis.findist(newpoints[0][0]//scale, newpoints[1][0]//scale), 1)
            nH= round(utilis.findist(newpoints[0][0]//scale, newpoints[2][0]//scale), 1)
            cv.arrowedLine(imgcont2, (newpoints[0][0][0], newpoints[0][0][1]), (newpoints[1][0][0], newpoints[1][0][1]),
                                    (255, 0, 255), 3, 8, 0, 0.05)
            cv.arrowedLine(imgcont2, (newpoints[0][0][0], newpoints[0][0][1]), (newpoints[2][0][0], newpoints[2][0][1]),
                            (255, 0, 255), 3, 8, 0, 0.05)
            x, y, w, h = obj[3]
            cv.putText(imgcont2, '{}cm'.format(nW), (x + 30, y - 10), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                        (255, 0, 255), 2)
            cv.putText(imgcont2, '{}cm'.format(nH), (x - 70, y + h // 2), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                        (255, 0, 255), 2)

            # print('polylines sucess')

    cv.imshow('Warped', imgcont2)

    img = cv.resize(img,(0,0), None,0.25,0.25)
    cv.imshow('Original', img)
    # cv.waitKey(1)

    if cv.waitKey(1) & 0xFF == 27:
        break
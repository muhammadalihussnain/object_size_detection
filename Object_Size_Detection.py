import cv2 as cv
import numpy as np
import utils
from time import sleep

cv.namedWindow("preview")
cap = cv.VideoCapture(1)
if cap.isOpened(): # try to get the first frame
    success, image = cap.read()
else:
    success = False




while True:
    success, image = cap.read()

    image, bbox,cornors =  utils.getContours(image, draw=True)
    image=  utils.warp(image, cornors)
    warpImage,wrpbbox,wrpedges  =   utils.getContours(image, draw=True, cannyT=[50, 50],max_area=1000)
    #warpImage=utils.warp(warpImage)(image,wrpedges)
    newpoints=utils.reorder(wrpedges)
    utils.findDis(newpoints)
    cv.imshow("preview", warpImage)




    key = cv.waitKey(1)
    if key == 27:  # exit on ESC#
        break

'''    cnt_image = utils.preProcessing(image)

    bounding_box = utils.getContours(cnt_image, draw=True)
    cv.rectangle(cnt_image,(bounding_box[0],bounding_box[1]),
                                     (bounding_box[0]+bounding_box[2],bounding_box[1]+bounding_box[3]),(0,255,0),3)
    cv.imshow("preview",cnt_image)'''








cap.release()
cv.destroyAllWindow()
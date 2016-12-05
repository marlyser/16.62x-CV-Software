#canny/hough sliders

import cv2
import numpy as np


img = cv2.imread('TA1_EV1_auto.png',0)

blur = cv2.GaussianBlur(img,(3,3),0)

ret,th1 = cv2.threshold(blur,50,255,cv2.THRESH_TOZERO)
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)


ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


#cv2.imshow('Thresholding', th1)

#k = cv2.waitKey(0) & 0xFF
#if k == 27:   # hit escape to quit
#	cv2.destoryAllWindows()



##HOUGH
## this function is needed for the createTrackbar step downstream


## read the experimental image
#img = cv2.imread('undistorted_TA1_EV1_auto.png', 0)
#orig = cv2.imread('TA1_EV1_auto.png', 0)
#img_copy = orig.copy()

## create trackbar for canny edge detection threshold changes
#cv2.namedWindow('T')

#def nothing(x):
    #img_copy = orig.copy()

## add lower and upper threshold slidebars to "canny"
#cv2.createTrackbar('threshold', 'T', 0, 255, nothing)

## Infinite loop until we hit the escape key on keyboard
#while(1):
	
    ## get current positions of four trackbars
	#t = cv2.getTrackbarPos('threshold','Hough')
	
	#lines = cv2.HoughLines(img,1,np.pi/180,t)
	#for rho,theta in lines[0]:
		#a = np.cos(theta)
		#b = np.sin(theta)
		#x0 = a*rho
		#y0 = b*rho
		#x1 = int(x0 + 1000*(-b))
		#y1 = int(y0 + 1000*(a))
		#x2 = int(x0 - 1000*(-b))
		#y2 = int(y0 - 1000*(a))

		#cv2.line(img_copy,(x1,y1),(x2,y2),(0,0,255),2)
	

    ## display images
	#cv2.imshow('Hough', img_copy)

	#k = cv2.waitKey(1) & 0xFF
	#if k == 27:   # hit escape to quit
		#break

#cv2.destroyAllWindows()

#~ #CANNY
#~ # this function is needed for the createTrackbar step downstream
def nothing(x):
    pass

# read the experimental image
#img = cv2.imread('TA1_EV1_auto.png', 0)

# create trackbar for canny edge detection threshold changes
cv2.namedWindow('canny')

# add ON/OFF switch to "canny"
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'canny', 0, 1, nothing)

# add lower and upper threshold slidebars to "canny"
cv2.createTrackbar('lower', 'canny', 0, 255, nothing)

# Infinite loop until we hit the escape key on keyboard
while(1):

    # get current positions of four trackbars
    lower = cv2.getTrackbarPos('lower', 'canny')
    s = cv2.getTrackbarPos(switch, 'canny')

    if s == 0:
        edges = blur
    else:
        rt, edges = cv2.threshold(blur,lower,255,cv2.THRESH_TOZERO)

    # display images
    #cv2.imshow('original', th2)
    cv2.imshow('canny', edges)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:   # hit escape to quit
        break

cv2.destroyAllWindows()

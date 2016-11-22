#Test for Canny edge detection parameters

import numpy as np
import cv2

def Canny_test(impath, sigma=0.20, thresholds=[]): #Thresholds is a list of other thresholds you might want to test (list of tuples)
	img = cv2.imread(impath, 0)
	
	med = np.median(img) #compute the median of pixel intensities
	
	lower = int(max(0, (1.0 - sigma)*med))
	upper = int(min(255,(1.0 + sigma)*med))
	print "lower: ", lower
	print "upper: ", upper
	
	canny_img = cv2.Canny(img, lower, upper)
	fname = "canny_%03.2f" %sigma + impath 
	cv2.imwrite(fname, canny_img)
	
	blurred = cv2.GaussianBlur(img, (3, 3), 0)
	cv2.imwrite("blurred_"+impath,blurred)
	med = np.median(blurred)
	lower = int(max(0, (1.0 - sigma)*med))
	upper = int(min(255,(1.0 + sigma)*med))
	print "blurred lower: ", lower
	print "blurred upper: ", upper
	canny_blurred = cv2.Canny(blurred, lower, upper)
	fname = "canny_%03.2f_" %sigma +impath 
	cv2.imwrite(fname, canny_blurred)
	
	if thresholds:
		for t in thresholds:
			canny_img  = cv2.Canny(img, t[0], t[1])
			fname = "canny_%d_%d.png" % (t[0], t[1])
			cv2.imwrite(fname, canny_img)		


def enhance_contrast(impath):
	img = cv2.imread(impath,0)
	equ = cv2.equalizeHist(img)
	cv2.imwrite("contrast_"+impath, equ)


enhance_contrast("calib_TA2_EV1.png")
Canny_test("contrast_calib_TA2_EV1.png", .25)

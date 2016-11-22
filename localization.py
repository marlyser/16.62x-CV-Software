#Computer Vision Localization of Lines in Test Articles

import numpy as np
import cv2

#Get horizontal and vertical angles from fiducial markers, calculate 45 degree angle

def image_undistort(raw_img, TA_name):
	#Camera parameters for undistortion given by camera calibration 
	mtx =  np.array([[1.54533000e+03,0.00000000e+00,8.41738145e+02],[0.00000000e+00,1.54502988e+03,6.19120876e+02],[0.00000000e+00, 0.00000000e+00,1.00000000e+00]]) 
	dist = np.array([[-2.99439288e-01,5.60252142e-02 ,3.06300269e-04,-1.73442748e-04,1.36723802e-01]])
	
	h,  w = raw_img.shape[:2]
	newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
	
	# undistort
	dst = cv2.undistort(raw_img, mtx, dist, None, newcameramtx)
	
	#crop image
	x,y,w,h = roi
	dst = dst[y:y+h, x:x+w]
	cv2.imwrite('undistorted_'+TA_name,dst)
	
	return dst
	
	
def image_preprocess(undist_img, TA_name):
	#Enhance Contrast (historgram equalization)
	equ = cv2.equalizeHist(undist_img)
	#equ = undist_img
	
	#Lessen noise (Gaussian Blur)
	blurred = cv2.GaussianBlur(equ, (3, 3), 0)
	#blurred = equ
	cv2.imwrite('preprocess_'+TA_name, blurred)
	
	return blurred
	


def Canny_detect(pre_img, TA_name):
	sigma = 0.33
	med = np.median(pre_img)
	
	lower = int(max(0, (1.0 - sigma)*med))
	upper = int(min(255,(1.0 + sigma)*med))
	
	canny_img = cv2.Canny(pre_img, lower, upper)
	cv2.imwrite('Canny_'+TA_name, canny_img)
	
	return canny_img
	



def Hough_lines(canny_img, TA_name, raw_img):
	lines = cv2.HoughLines(canny_img,1,np.pi/180,150)
	for i in range(len(lines)):
		rho = lines[i][0][0]
		theta = lines[i][0][1]
		e = 0.1 #radians
		if rho < 0+e or np.pi/4-e < rho < np.pi/4+e or np.pi/2-e < rho < np.pi/2+e:
			a = np.cos(theta)
			b = np.sin(theta)
			x0 = a*rho
			y0 = b*rho
			x1 = int(x0 + 1000*(-b))
			y1 = int(y0 + 1000*(a))
			x2 = int(x0 - 1000*(-b))
			y2 = int(y0 - 1000*(a))

			cv2.line(raw_img,(x1,y1),(x2,y2),(0,0,255),2)
	
	cv2.imwrite('Hough_'+TA_name, raw_img)
	
	return canny_img
	

def localization(img_path):
	print "Localizing..."
	
	raw_img = cv2.imread(img_path, 0)
	
	undst_img = image_undistort(raw_img, img_path)
	
	pre_img = image_preprocess(undst_img, img_path)
	
	canny_img = Canny_detect(pre_img, img_path)
	
	hough_img = Hough_lines(canny_img, img_path, undst_img)
	
	print "Localization done"
	
localization("TA1_EV1_auto.png")

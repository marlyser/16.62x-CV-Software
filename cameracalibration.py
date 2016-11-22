#Camera Calibration

import numpy as np
import cv2
import glob


def init_cameracalibrate(raw_image):
	# termination criteria
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

	# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
	objp = np.zeros((9*9,3), np.float32)
	objp[:,:2] = np.mgrid[0:9,0:9].T.reshape(-1,2)

	# Arrays to store object points and image points from all the images.
	objpoints = [] # 3d point in real world space
	imgpoints = [] # 2d points in image plane.

	images = glob.glob('calib/*.png')

	for fname in images:
		img = cv2.imread(fname)
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

		# Find the chess board corners
		ret, corners = cv2.findChessboardCorners(gray, (9,9),None)

		# If found, add object points, image points (after refining them)
		if ret == True:
			objpoints.append(objp)

			corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
			imgpoints.append(corners2)

			# Draw and display the corners
			img = cv2.drawChessboardCorners(img, (9,9), corners2,ret)
			cv2.imshow('img',img)
			cv2.waitKey(500)

	cv2.destroyAllWindows()
	
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None) #These are the camera properties to be saved
	print "Camera matrix: ", mtx
	print "Distortion Coeff: ", dist
	
	


#init_cameracalibrate('TA1_EV1_auto.png')

mtx =  np.array([[1.54533000e+03,0.00000000e+00,8.41738145e+02],[0.00000000e+00,1.54502988e+03,6.19120876e+02],[0.00000000e+00, 0.00000000e+00,1.00000000e+00]])
 
dist = np.array([[-2.99439288e-01,5.60252142e-02 ,3.06300269e-04,-1.73442748e-04,1.36723802e-01]])



def image_undistort(path_to_image):
	
	img = cv2.imread(path_to_image)
	h,  w = img.shape[:2]
	newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
	
	# undistort
	dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

	# crop the image
	x,y,w,h = roi
	dst = dst[y:y+h, x:x+w]
	cv2.imwrite('calib_TA2_EV1.png',dst)

image_undistort('TA2_EV1.png')

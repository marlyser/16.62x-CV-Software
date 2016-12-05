#Computer Vision Localization of Lines in Test Articles

import numpy as np
import cv2
import math

#Get horizontal and vertical angles from fiducial markers, calculate 45 degree angle

fids = [(1330, 846), (1336, 1040), (947, 1060)]

zero = np.arctan2(abs(fids[1][1] - fids[2][1]),abs(fids[1][0] - fids[2][0]))

true_line = [(890.48664495114008, 0.0), (943.0208469055375, 1120.0)]

conf_factor = 0.26 #[mm/pix]


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
	#equ = cv2.equalizeHist(undist_img)
	equ = undist_img
	kernel = np.ones((5,5),np.uint8)
	#Lessen noise (Gaussian Blur)
	blurred = cv2.GaussianBlur(equ, (3, 3), 0)
	#blurred = equ
	
	ret,blurred = cv2.threshold(blurred,60,255,cv2.THRESH_TRUNC)
	
	#th2 = cv2.adaptiveThreshold(undist_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
	
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
	
def get_endpoints(rho, theta):
	endpoints = []
	
	slope = -1*np.cos(theta)/np.sin(theta)
	intercept = rho/np.sin(theta)
	
	endpoints.append(slope)
	endpoints.append(intercept)
	
	#Top
	A1 = np.array([[-slope, 1], [0, 1]])
	B1 = np.array([intercept, 0])
	try:
		x1,y1 = np.linalg.solve(A1, B1)
		if 0 <= x1<= 1553 and 0<=y1 <=1120:
			endpoints.append((x1,y1))
	except np.linalg.linalg.LinAlgError:
		pass
	
	#Bottom
	A2 = np.array([[-slope, 1], [0, 1]])
	B2 = np.array([intercept, 1120])
	try:
		x2,y2 = np.linalg.solve(A2, B2)
		if 0 <= x2<= 1553 and 0<=y2 <=1120:
			endpoints.append((x2,y2))
	except np.linalg.linalg.LinAlgError:
		pass
	
	#Left
	A3 = np.array([[-slope, 1], [1, 0]])
	B3 = np.array([intercept, 0])
	try:
		x3,y3 = np.linalg.solve(A3, B3)
		if 0 <= x3<= 1553 and 0<=y3 <=1120:
			endpoints.append((x3,y3))
	except np.linalg.linalg.LinAlgError:
		pass
	
	#Right
	A4 = np.array([[-slope, 1], [1, 0]])
	B4 = np.array([intercept, 1553])
	try:
		x4,y4 = np.linalg.solve(A4, B4)
		if 0 <= x4<= 1553 and 0<=y4 <=1120:
			endpoints.append((x4,y4))
	except np.linalg.linalg.LinAlgError:
		pass
	
	return endpoints
	
def line_Isequal(l1, l2): #lines given as (rho, theta)
	if abs(l1[0]-l2[0]) >= max(abs(l1[0]*0.1), abs(l2[0]*.1)):
		return False
	
	if abs(l1[1]-l2[1]) >= math.radians(3):
		return False
	
	return True


def cluster_lines(all_lines):
	n = 0
	d = {}
	c_lines = []
	for line in all_lines:
		if not d:
			d[n] = [line]
		else:
			for k in d.keys():
				if line_Isequal(line, d[k][0]):
					d[k].append(line)
					break
				elif k == len(d)-1:
					d[k+1] = [line]
					break

	for cluster in d.values():
		c_lines.append((np.mean([i[0] for i in cluster]), np.mean([i[1] for i in cluster])))
	return c_lines
		
	


def Hough_lines(canny_img, TA_name, raw_img):
	t = 100
	all_lines = set([])
	raw_c = raw_img.copy()
	lines = cv2.HoughLines(canny_img,1,np.pi/180,t)
	for i in range(len(lines)):
		rho = lines[i][0][0]
		theta = lines[i][0][1]
		all_lines.add((rho,theta))
		#if e == 20:#rho < zero+e or np.pi/4-e+zero < rho < np.pi/4+e+zero or np.pi/2-e+zero < rho < np.pi/2+e+zero or np.pi*2-e < rho < np.pi*2:
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
	
	c_lines = cluster_lines(all_lines)

	endpoints = []
	for (rho,theta) in c_lines:
		print (rho,theta)
		a = np.cos(theta)
		b = np.sin(theta)
		x0 = a*rho
		y0 = b*rho
		x1 = int(x0 + 1000*(-b))
		y1 = int(y0 + 1000*(a))
		x2 = int(x0 - 1000*(-b))
		y2 = int(y0 - 1000*(a))
		cv2.line(raw_c,(x1,y1),(x2,y2),(0,0,255),2)
		
		endpoints.append(get_endpoints(rho,theta))
		
	cv2.imwrite('Clustered_'+TA_name, raw_c)
	
	
	
	return endpoints
	
def filter_endpoints(endpoints):
	top = [y for y in endpoints if y[2][1] == 0]
	left = [x for x in endpoints if x[2][0] == 0]
	
	if len(top) > 3 or len(left) > 3:
		return "Fail: Too many edges detected"
	else:
		if len(top) == 3:
			top.sort(key=lambda x: x[2][0])
			return top[1]
		elif len(left) == 3 or len(left) == 2:
			left.sort(key = lambda y: y[2][1]).reverse()
			return left[1]
		else:
			return "Fail: Too few edges detected"

def draw_lines(img, true_line, exp_line, TA_name):
	cv2.line(img,(int(true_line[0][0]),int(true_line[0][1])) ,(int(true_line[1][0]),int(true_line[1][1])),(255,0,0),2)
	cv2.line(img,(int(exp_line[0][0]),int(exp_line[0][1])),(int(exp_line[1][0]),int(exp_line[1][1])),(0,0,255),2)
	
	cv2.imwrite('Compare'+TA_name, img)


def pass_fail(true_line, exp_line):
	mm_1 = 1/conf_factor
	
	print true_line
	m = (true_line[1][1] - true_line[0][1])/(true_line[1][0] - true_line[0][0])
	alpha = np.arctan(1/m)
	theta = np.pi/2 - alpha
	x = mm_1/np.sin(theta)
	y = mm_1/np.sin(alpha)
	
	dist1 = np.linalg.norm(np.array(true_line[0]) - np.array(exp_line[0]))
	dist2 = np.linalg.norm(np.array(true_line[1]) - np.array(exp_line[1]))
	
	print dist1
	print dist2
	print x, y
	
	if exp_line[0][1] == 1120 or exp_line[0][1] == 0:
		if dist1 < x:
			return 'Pass'
		else:
			return 'Fail'
			
	if exp_line[0][0] == 0 or exp_line[0][0] == 1553:
		if dist2 < y:
			return 'Pass'
		else:
			return 'Fail'
			
	if exp_line[1][1] == 1120 or exp_line[3][1] == 0:
		if dist1 < x:
			return 'Pass'
		else:
			return 'Fail'
			
	if exp_line[1][0] == 0 or exp_line[1][0] == 1553:
		if dist2 < y:
			return 'Pass'
		else:
			return 'Fail'
	
	


	
	
	
	
	

def localization(img_path):
	print "Localizing..."
	
	raw_img = cv2.imread(img_path, 0)
	
	undst_img = image_undistort(raw_img, img_path)
	
	pre_img = image_preprocess(undst_img, img_path)
	
	undst_img_copy = undst_img.copy()
	
	canny_img = Canny_detect(pre_img, img_path)
	
	all_endpoints = Hough_lines(canny_img, img_path, undst_img_copy)
	
	exp_line = filter_endpoints(all_endpoints)
	
	if type(exp_line) is str:
		print exp_line
		return
	
	draw_lines(undst_img,true_line, (exp_line[2], exp_line[3]), img_path)
	
	result = pass_fail(true_line, (exp_line[2], exp_line[3]))
	print result
	
	print "Localization done"

localization('TA1_EV1_auto.png')


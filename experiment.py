#Runs through full experimental procedure
#0) Image Selection 1) Image undistortion 2) Camera Calibration 3) Truth Data Collection 4) Localization 5) Data Storage

#Undistort image, get pixel locations for fiducials, store pix-to-mm conversion, get horizontal and vertical transformations
import time, datetime
import numpy as np
import cv2
import Tkinter
from PIL import Image, ImageTk
from scipy import stats
import math, csv

data = []
filename = 'data_marlyse.csv'
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
data.append(st)

TA = 40
folder = "/home/mreeves/16.62x/Results/Test Article " + str(TA)+'/'
TA_name = 'TA40_EV5.png'
raw_img = cv2.imread(folder+TA_name, 0)
data.append(TA_name)

#1) IMAGE UNDISTORTION
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
	cv2.imwrite(folder+'undistorted_'+TA_name,dst)
	
	return dst
undst = image_undistort(raw_img, TA_name)
undst_copy = undst.copy()
undst_copy2 = undst.copy()

#2) CAMERA CALIBRATION & TRUTH DATA COLLECTION
global mouseX,mouseY

def fiducial_collect(img, fid_num, fid_pts):
	name = 'Collect Fiducials '+ str(fid_num)
	cv2.namedWindow(name, cv2.WINDOW_NORMAL)
	
	def draw_circle(event,x,y,flags,param):
		if event == cv2.EVENT_LBUTTONDBLCLK:
			cv2.circle(img,(x,y),1,(255,0,0),-1)
			mouseX,mouseY = x,y
			fid_pts[fid_num-1][0] = (mouseX, mouseY)
	
	cv2.setMouseCallback(name, draw_circle)
	while(1):
		cv2.imshow(name, img)
		if cv2.waitKey(20) & 0xFF == 27:
			break
	#cv2.destroyAllWindows()
	
def points_collect(img, ln_pts):
	name = 'Collect Points'
	cv2.namedWindow(name, cv2.WINDOW_NORMAL)
	
	def draw_circle(event,x,y,flags,param):
		if event == cv2.EVENT_LBUTTONDBLCLK:
			cv2.circle(img,(x,y),1,(255,0,0),-1)
			mouseX,mouseY = x,y
			ln_pts.add((mouseX, mouseY))
	
	cv2.setMouseCallback(name, draw_circle)
	while(1):
		cv2.imshow(name, img)
		if cv2.waitKey(20) & 0xFF == 27:
			break
 
class GUIapp_tk(Tkinter.Tk):
	def __init__(self, parent):
		Tkinter.Tk.__init__(self, parent)
		self.parent = parent
		self.initialize()
		self.img = undst_copy
		self.fid_pts = [[0],[0],[0]]
		self.img2 = undst_copy2
		self.line_points = set([])
		
	def initialize(self):
		self.grid()
		
		button1 = Tkinter.Button(self, text="Collect Fiducial 1", command = self.ButtonClick1)
		button1.grid(column=0, row=2)
		
		button2 = Tkinter.Button(self, text="Collect Fiducial 2", command = self.ButtonClick2)
		button2.grid(column=0, row=3)
		
		button3 = Tkinter.Button(self, text="Collect Fiducial 3", command = self.ButtonClick3)
		button3.grid(column=0, row=4)
		
		button4 = Tkinter.Button(self, text="Done!", command = self.ButtonClickDone)
		button4.grid(column=1, row=5)
		
		button5 = Tkinter.Button(self, text="Reset_Fiducials", command = self.ButtonReset)
		button5.grid(column=0, row=1)
		
		button6 = Tkinter.Button(self, text="Collect Truth Data Points", command = self.ButtonClick6)
		button6.grid(column=2, row=2)
		
		button7 = Tkinter.Button(self, text="Reset_Truth", command = self.ButtonReset2)
		button7.grid(column=2, row=1)
		
	def ButtonClick1(self):
		cv2.destroyAllWindows()
		fiducial_collect(self.img, 1, self.fid_pts)
		
	def ButtonClick2(self):
		cv2.destroyAllWindows()
		fiducial_collect(self.img, 2, self.fid_pts)
		
	def ButtonClick3(self):
		cv2.destroyAllWindows()
		fiducial_collect(self.img, 3, self.fid_pts)
	
	def ButtonClickDone(self):
		cv2.destroyAllWindows()
		self.destroy()
		
	def ButtonReset(self):
		self.img = undst.copy()
		self.fid_pts = [[0],[0],[0]]
		
	def ButtonClick6(self):
		cv2.destroyAllWindows()
		points_collect(self.img2, self.line_points)
	
	def ButtonClickDone(self):
		cv2.destroyAllWindows()
		self.destroy()
		
	def ButtonReset2(self):
		cv2.destroyAllWindows()
		self.img2 = undst.copy()
		self.line_points = set([])
		
#Run Calibration/Truth Data GUI	
if __name__ == "__main__":
		app = GUIapp_tk(None)
		app.title('Experiment GUI')
		app.mainloop()

fiducial_points = app.fid_pts


def get_pixtomm(points):
	horz_dist = 25.4*4 #[mm]
	vert_dist = 25.4*2 #[mm]
	
	#Vertical is 1 and 2
	conv1 = vert_dist/np.linalg.norm(np.array(points[0][0])-np.array(points[1][0]))
	#print conv1
	#Horizontal is 2 and 3
	conv2 = horz_dist/np.linalg.norm(np.array(points[1][0])-np.array(points[2][0]))
	#print conv2

	pixtomm = (conv1+conv2)/2
	return pixtomm

pixtomm = get_pixtomm(app.fid_pts)
print pixtomm
data.append(str(pixtomm))


#p = set([(917, 257), (921, 448), (931, 930), (923, 598), (928, 788)])
line_points = app.line_points
print line_points

def get_true_edge(points, img):
	unzipped = zip(*points)
	
	x = np.array(unzipped[0])
	y = np.array(unzipped[1])
	
	slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
	
	endpoints = []
	
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


	#~ cv2.line(img,(int(endpoints[0][0]),int(endpoints[0][1])),(int(endpoints[1][0]),int(endpoints[1][1])),(0,0,255),2)
	
	
	#~ cv2.imwrite('truth_'+ TA_name, img)
	
	return (slope,intercept, endpoints[0], endpoints[1])

true_line = get_true_edge(line_points, undst_copy)
data.append(str((true_line[2], true_line[3])))
data.append(str(np.linalg.norm(np.array(true_line[2]) - np.array(true_line[3]))))

#4) LOCALIZATION

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
	
	cv2.imwrite(folder+'preprocess_'+TA_name, blurred)
	
	return blurred
	
def Canny_detect(pre_img, TA_name):
	sigma = 0.33
	med = np.median(pre_img)
	
	lower = int(max(0, (1.0 - sigma)*med))
	upper = int(min(255,(1.0 + sigma)*med))
	
	canny_img = cv2.Canny(pre_img, lower, upper)
	
	
	cv2.imwrite(folder+'Canny_'+TA_name, canny_img)
	
	return canny_img
	
def get_endpoints(rho, theta):
	endpoints = []
	print rho, theta
	if theta == 2*np.pi or theta == np.pi or theta == 0:
		slope = -1
		intercept = rho
	else:
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
	if abs(abs(l1[0])-abs(l2[0])) >= 100: #max(abs(l1[0]*0.1), abs(l2[0]*.1)):
		return False
	
	if abs(l1[1]-l2[1]) >= math.radians(5):
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
	if lines is None:
		return 'Fail: No edges detected'
	for i in range(len(lines)):
		rho = lines[i][0][0]
		theta = lines[i][0][1]
		if theta == 0:
			theta = np.pi
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
			
	cv2.imwrite(folder+'Hough_'+TA_name, raw_img)
	
	c_lines = cluster_lines(all_lines)

	endpoints = []
	for (rho,theta) in c_lines:
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
		
	cv2.imwrite(folder+'Clustered_'+TA_name, raw_c)
	
	
	
	return endpoints
	
def filter_endpoints(endpoints):
	if len(endpoints) > 5:
		return "Fail: Too many edges detected"
	top = [y for y in endpoints if y[2][1] == 0 or y[2][1] == 1120]
	left = [x for x in endpoints if x[2][0] == 0 or x[2][0] == 1553]
	
	if len(top) == 3:
		top.sort(key=lambda x: x[2][0])
		top_l = [np.linalg.norm(np.array(x[2])-np.array(x[3])) for x in top]
		if top_l.index(max(top_l)) != 1:
			return top[top_l.index(max(top_l))]
		return top[1]
	elif len(left) == 3 or len(left) == 2:
		left.sort(key = lambda y: y[2][1])
		left.reverse()
		left_l = [np.linalg.norm(np.array(x[2])-np.array(x[3])) for x in left]
		if left_l.index(max(left_l)) != 1:
			return left[left_l.index(max(left_l))]
		return left[1]
	else:
		return "Fail: Too few edges detected"

def draw_lines(img, true_line, exp_line, TA_name):
	cv2.line(img,(int(true_line[0][0]),int(true_line[0][1])) ,(int(true_line[1][0]),int(true_line[1][1])),(255,0,0),2)
	cv2.line(img,(int(exp_line[0][0]),int(exp_line[0][1])),(int(exp_line[1][0]),int(exp_line[1][1])),(0,0,255),2)
	
	cv2.imwrite(folder+'Compare'+TA_name, img)

def pass_fail(true_line, exp_line):
	mm_1 = 1/pixtomm
	
	m = true_line[0]
	alpha = np.arctan(1/m)
	theta = np.pi/2 - alpha
	x = abs(mm_1/np.sin(theta))
	y = abs(mm_1/np.sin(alpha))
	

	
	dist1 = np.linalg.norm(np.array(true_line[2]) - np.array(exp_line[2]))
	dist2 = np.linalg.norm(np.array(true_line[3]) - np.array(exp_line[3]))
	
	if true_line[2][1] == 1120 or true_line[2][1] == 0:
		if dist1 < x:
			return 'Pass'
		else:
			return 'Fail'
			
	if true_line[2][0] == 0 or true_line[2][0] == 1553:
		if dist2 < y:
			return 'Pass'
		else:
			return 'Fail'
			
	if true_line[3][1] == 1120 or true_line[3][1] == 0:
		if dist1 < x:
			return 'Pass'
		else:
			return 'Fail'
			
	if true_line[3][0] == 0 or true_line[3][0] == 1553:
		if dist2 < y:
			return 'Pass'
		else:
			return 'Fail'

def swept_area(true_line, exp_line):
	A = np.array([[-true_line[0], -exp_line[0]], [1, 1]])
	B = np.array([true_line[1], exp_line[1]])
	try:
		x,y = np.linalg.solve(A, B)
	except np.linalg.linalg.LinAlgError:
		pass
	
	a1 = np.linalg.norm(np.array(true_line[2]) - np.array(exp_line[2]))
	b1 = np.linalg.norm(np.array((x,y)) - np.array(exp_line[2]))
	c1 = np.linalg.norm(np.array((x,y)) - np.array(true_line[2]))
	s1 = (a1+b1+c1)/2
	area1 = (s1*(s1-a1)*(s1-b1)*(s1-c1)) ** 0.5
	
	a2 = np.linalg.norm(np.array(true_line[3]) - np.array(exp_line[3]))
	b2 = np.linalg.norm(np.array((x,y)) - np.array(exp_line[3]))
	c2 = np.linalg.norm(np.array((x,y)) - np.array(true_line[3]))
	s2 = (a2+b2+c2)/2
	area2 = (s2*(s2-a2)*(s2-b2)*(s2-c2)) ** 0.5	
	
	return (area1 + area2)*pixtomm**2		

def localization(undst_img):
	print "Localizing..."
	
	pre_img = image_preprocess(undst, TA_name)
	
	undst_img_copy = undst.copy()
	
	canny_img = Canny_detect(pre_img, TA_name)
	
	all_endpoints = Hough_lines(canny_img, TA_name, undst_img_copy)
	
	if type(all_endpoints) is str:
		print all_endpoints
		data.append(all_endpoints)
		return
	
	exp_line = filter_endpoints(all_endpoints)
	
	
	
	if type(exp_line) is str:
		print exp_line
		data.append(exp_line)
		data.append('0')
		return
	
	data.append(str((exp_line[2], exp_line[3])))
	data.append(str(np.linalg.norm(np.array(exp_line[2]) - np.array(exp_line[3]))))
	
	draw_lines(undst_img,(true_line[2], true_line[3]), (exp_line[2], exp_line[3]), TA_name)
	
	result = pass_fail(true_line, exp_line)
	data.append(result)
	
	print result
	
	area_swept = swept_area(true_line, exp_line)
	data.append(area_swept)
	
	print "Localization done"

result = localization(undst)

#5) DATA STORAGE

with open(filename, 'a') as f:
	writer = csv.writer(f)
	writer.writerow(data)





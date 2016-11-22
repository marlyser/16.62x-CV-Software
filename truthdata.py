#Collect Truth Data Line

import numpy as np
import cv2
import Tkinter
from scipy import stats
from PIL import Image, ImageTk

TA_name = 'TA1_EV1_auto.png'
TA_num = int(TA_name[2])
EV_num = int(TA_name[6])

def image_undistort(TA_name):
	raw_img = cv2.imread(TA_name, 0)
	
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

undst = image_undistort(TA_name)
undst_copy = undst.copy()


global mouseX,mouseY

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

class truthapp_tk(Tkinter.Tk):
	def __init__(self, parent):
		Tkinter.Tk.__init__(self, parent)
		self.parent = parent
		self.initialize()
		self.img = undst_copy
		self.line_points = set([])
		
	def initialize(self):
		self.grid()
		
		button1 = Tkinter.Button(self, text="Collect Truth Data Points", command = self.ButtonClick1)
		button1.grid(column=0, row=2)
		
		button4 = Tkinter.Button(self, text="Done!", command = self.ButtonClickDone)
		button4.grid(column=0, row=5)
		
		button5 = Tkinter.Button(self, text="Reset", command = self.ButtonReset)
		button5.grid(column=0, row=1)
		
	def ButtonClick1(self):
		cv2.destroyAllWindows()
		points_collect(self.img, self.line_points)
	
	def ButtonClickDone(self):
		cv2.destroyAllWindows()
		self.destroy()
		
	def ButtonReset(self):
		cv2.destroyAllWindows()
		self.img = undst.copy()
		self.line_points = set([])
		
	
if __name__ == "__main__":
		app = truthapp_tk(None)
		app.title('Collect Truth Data')
		app.mainloop()


print app.line_points

p = set([(916, 536), (908, 352), (901, 226), (910, 445), (896, 116)])


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
		endpoints.append((x1,y1))
	except np.linalg.linalg.LinAlgError:
		pass
	
	#Bottom
	A2 = np.array([[-slope, 1], [0, 1]])
	B2 = np.array([intercept, 1120])
	try:
		x2,y2 = np.linalg.solve(A2, B2)
		endpoints.append((x2,y2))
	except np.linalg.linalg.LinAlgError:
		pass
	
	#Left
	A3 = np.array([[-slope, 1], [1, 0]])
	B3 = np.array([intercept, 0])
	try:
		x3,y3 = np.linalg.solve(A3, B3)
		endpoints.append((x3,y3))
	except np.linalg.linalg.LinAlgError:
		pass
	
	#Right
	A4 = np.array([[-slope, 1], [1, 0]])
	B4 = np.array([intercept, 1553])
	try:
		x4,y4 = np.linalg.solve(A4, B4)
		endpoints.append((x4,y4))
	except np.linalg.linalg.LinAlgError:
		pass


	cv2.line(img,(int(endpoints[0][0]),int(endpoints[0][1])),(int(endpoints[1][0]),int(endpoints[1][1])),(0,0,255),2)
	
	
	cv2.imwrite('truth_'+ TA_name, img)
	
	return endpoints

truth_data = get_true_edge(p, undst_copy)

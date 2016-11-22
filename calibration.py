#Undistort image, get pixel locations for fiducials, store pix-to-mm conversion, get horizontal and vertical transformations

import numpy as np
import cv2
import Tkinter
from PIL import Image, ImageTk

TA_name = 'TA1_EV1_auto.png'
TA_num = int(TA_name[2])
EV_num = int(TA_name[6])

horz_dist = 25.4*4 #[mm]
vert_dist = 25.4*2 #[mm]

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
 
class calibapp_tk(Tkinter.Tk):
	def __init__(self, parent):
		Tkinter.Tk.__init__(self, parent)
		self.parent = parent
		self.initialize()
		self.img = undst_copy
		self.fid_pts = [[0],[0],[0]]
		
	def initialize(self):
		self.grid()
		
		button1 = Tkinter.Button(self, text="Collect Fiducial 1", command = self.ButtonClick1)
		button1.grid(column=0, row=2)
		
		button2 = Tkinter.Button(self, text="Collect Fiducial 2", command = self.ButtonClick2)
		button2.grid(column=0, row=3)
		
		button3 = Tkinter.Button(self, text="Collect Fiducial 3", command = self.ButtonClick3)
		button3.grid(column=0, row=4)
		
		button4 = Tkinter.Button(self, text="Done!", command = self.ButtonClickDone)
		button4.grid(column=0, row=5)
		
		button5 = Tkinter.Button(self, text="Reset", command = self.ButtonReset)
		button5.grid(column=0, row=1)
		
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
		
	
if __name__ == "__main__":
		app = calibapp_tk(None)
		app.title('Calibrate')
		app.mainloop()

print app.fid_pts

def get_pixtomm(points):
	#Vertical is 1 and 2
	conv1 = vert_dist/np.linalg.norm(np.array(points[0][0])-np.array(points[1][0]))
	print conv1
	#Horizontal is 2 and 3
	conv2 = horz_dist/np.linalg.norm(np.array(points[1][0])-np.array(points[2][0]))
	print conv2

	pixtomm = (conv1+conv2)/2
	return pixtomm

pixtomm = get_pixtomm(app.fid_pts)
print pixtomm


	

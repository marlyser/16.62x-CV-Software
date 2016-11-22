#This is a class for each evaluation

import numpy
import cv2


class Evaluation:
	
	
	def _init_(self, test_article, evalut, raw_image):
		self.test_article = test_article
		self.evalut = evalut
		self.raw = raw
		self.undistorted = None
		self.pixtomm = None
		self.Canny = None
		self.Hough = {}
		
	#def save_data(self, 
	
	
	#def save_image

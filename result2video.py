#!/usr/bin/env python 
import os
import cv2
import numpy as np

def result_add_frames(resultNum, vehicleTrack):
	# print("Adding frames for", resultNum)
	# draw rectangles on the original image
	imageName = 'image/'+str(resultNum)+'.jpg'
	img = cv2.imread(imageName)
	# for id_vehicle, x, y in vehicleTrack:
	for x, y in vehicleTrack:
		cv2.rectangle(img, (x-16,y-16), (x+16,y+16), (0,255,255), 1)
	cv2.imwrite('result/' + str(resultNum) + '.jpg', img)

def result_to_video():
	img_root = 'result/'
	#Edit each frame's appearing time!  
	fps = 30
	fourcc = cv2.VideoWriter_fourcc(*"MJPG")  
	videoWriter = cv2.VideoWriter("result.mp4", fourcc, fps, (660,720))  
	  
	im_names = os.listdir(img_root)  
	for im_name in range(len(im_names)):  
		frame = cv2.imread(img_root + str(im_name+1) + '.jpg')  
		print(img_root + str(im_name+1) + '.jpg') 
		videoWriter.write(frame)  
		  
	videoWriter.release()  


class VehiclePosition():
	def __init__(self):
		print('Vehicle Position Class')
		self.VehicleBuf = list()

	def update_vehicle_buffer(self, truePoint, delPre=False):
		if delPre:
			del self.VehicleBuf[0]
		self.VehicleBuf.append(truePoint)

	def get_vehicles(self, truePoint):
		vehiclePosition = list()
		for x_center, y_center in truePoint:
			flag = False
			numTemp = 0
			for points in self.VehicleBuf:
				for x,y in points:
					if (x_center-x)**2 + (y_center-y)**2 < 500:
						numTemp += 1
					if numTemp >= 3:
						flag = True
						vehiclePosition.append([x_center,y_center])
						break
				if flag:
					break
		return vehiclePosition



if __name__ == "__main__":
	# result_add_frames(180)


	result_to_video()
import cv2
import numpy as np
import time

class Data():
	def __init__(self, batchSize=32):
		print('DATA')
		self.imageIndex = 1
		self.groundtruthIndex = 1
		self.batchSize = batchSize

	def get_one_train_GT(self):
		oneGT = np.zeros((720,660))
		GTName = 'groundtruth/'+ str(self.groundtruthIndex*6) +'.png'
		self.groundtruthIndex += 1
		if self.groundtruthIndex > 150:
			self.groundtruthIndex = 1
		groundTruth = cv2.cvtColor(cv2.imread(GTName), cv2.COLOR_RGB2GRAY)
		x,y = groundTruth.shape
		for i in range(720):
			for j in range(660):
				if groundTruth[int(i),int(j)] != 0:
					oneGT[i,j] = 1
		# np.save(str((self.groundtruthIndex-1)*6) + 'npy', oneGT)
		return oneGT

	def get_one_train_image(self):
		imageName = 'image/'+ str(self.imageIndex*6) +'.jpg' 
		image = cv2.cvtColor(cv2.imread(imageName), cv2.COLOR_RGB2GRAY).reshape((720,660,1)).astype(np.float32)
		self.imageIndex += 1
		if self.imageIndex > 150:
			self.imageIndex = 1
		return image/127.5-1

	def get_next_batch_image(self):
		# start_time = time.time()
		# print('Getting a batch of images.')
		imageBatch = np.zeros((self.batchSize,720,660,1))
		for i in range(self.batchSize):
			imageBatch[i,:,:,:] = self.get_one_train_image()
		# print('It took', time.time()-start_time, 's')
		return imageBatch

	def get_next_batch_train_data(self):
		# start_time = time.time()
		# print('Getting a batch of training data.')
		imageBatch = np.zeros((self.batchSize,720,660,1))
		gtBatch = np.zeros((self.batchSize,720,660))
		for i in range(self.batchSize):
			imageBatch[i,:,:,:] = self.get_one_train_image()
			gtBatch[i,:,:] = self.get_one_train_GT()
		# print('It took', time.time()-start_time, 's')
		return imageBatch, gtBatch


if __name__ == "__main__":
	data = Data(batchSize=1)
	for i in range(150):
		data.get_one_train_GT()

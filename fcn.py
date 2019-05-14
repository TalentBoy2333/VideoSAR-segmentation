import tensorflow as tf 
import numpy as np 
import cv2
from data import *
from result2video import *

class FCN():
	def __init__(self, batchSize):
		print('Full conv Net')
		self.batchSize = batchSize
		self.learningRate = 0.0001

	def weight_variable(self, shape, name=None):
		initial = tf.truncated_normal(shape,stddev = 0.1,dtype = tf.float32,name = name)
		return tf.Variable(initial)

	def bias_variable(self, shape, name=None):
		initial = tf.constant(0.1,shape = shape,dtype = tf.float32,name = name)
		return tf.Variable(initial)

	def conv2d(self, x, W):
		return tf.nn.conv2d(x,W,strides = [1,1,1,1],padding = 'SAME')

	def conv2d_2x2(self, x, W):
		return tf.nn.conv2d(x,W,strides = [1,2,2,1],padding = 'SAME')

	def max_pool_2x2(self, x):
		return tf.nn.max_pool(x,ksize = [1,2,2,1],strides = [1,2,2,1],padding = 'SAME')

	def deconv2d(self, x, W, outputSize, featureMapSize):
		return tf.nn.conv2d_transpose(x, W, output_shape = [self.batchSize, outputSize[0], outputSize[1], featureMapSize], strides = [1, 2, 2, 1], padding = 'SAME')

	def build(self):
		self.images = tf.placeholder(tf.float32, [None, 720, 660, 1])
		self.ys = tf.placeholder(tf.float32, [None, 720, 660])
		# self.y_truth = tf.reshape(self.ys, [-1,3])
		# CNN layer 1
		self.W_conv1_1 = self.weight_variable([3,3,1,16],"wc1_1") 
		self.b_conv1_1 = self.bias_variable([16],"bc1_1")
		self.output_conv1_1 = self.conv2d(self.images,self.W_conv1_1) + self.b_conv1_1 # (720, 660, 1 Image) ==> (720, 660, 16 featureMaps)
		self.h_conv1_1 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(self.output_conv1_1, \
				decay=0.9, updates_collections=None, epsilon=1e-5, center=True, scale=True, is_training=True))

		self.W_conv1_2 = self.weight_variable([3,3,16,16],"wc1_2") 
		self.b_conv1_2 = self.bias_variable([16],"bc1_2")
		self.output_conv1_2 = self.conv2d(self.h_conv1_1,self.W_conv1_2) + self.b_conv1_2 # (720, 660, 16 featureMaps) ==> (720, 660, 16 featureMaps)
		self.h_conv1_2 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(self.output_conv1_2, \
				decay=0.9, updates_collections=None, epsilon=1e-5, center=True, scale=True, is_training=True))

		self.W_conv1_3 = self.weight_variable([3,3,16,16],"wc1_3") 
		self.b_conv1_3 = self.bias_variable([16],"bc1_3")
		self.output_conv1_3 = self.conv2d_2x2(self.h_conv1_2,self.W_conv1_3) + self.b_conv1_3 # (720, 660, 16 featureMaps) ==> (360, 330, 16 featureMaps)
		self.h_conv1_3 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(self.output_conv1_3, \
				decay=0.9, updates_collections=None, epsilon=1e-5, center=True, scale=True, is_training=True))
		# CNN layer 2
		self.W_conv2_1 = self.weight_variable([3,3,16,32],"wc2_1") 
		self.b_conv2_1 = self.bias_variable([32],"bc2_1")
		self.output_conv2_1 = self.conv2d(self.h_conv1_3,self.W_conv2_1) + self.b_conv2_1 # (360, 330, 16 featureMaps) ==> (360, 330, 32 featureMaps)
		self.h_conv2_1 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(self.output_conv2_1, \
				decay=0.9, updates_collections=None, epsilon=1e-5, center=True, scale=True, is_training=True))

		self.W_conv2_2 = self.weight_variable([3,3,32,32],"wc2_2") 
		self.b_conv2_2 = self.bias_variable([32],"bc2_2")
		self.output_conv2_2 = self.conv2d(self.h_conv2_1,self.W_conv2_2) + self.b_conv2_2 # (360, 330, 32 featureMaps) ==> (360, 330, 32 featureMaps)
		self.h_conv2_2 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(self.output_conv2_2, \
				decay=0.9, updates_collections=None, epsilon=1e-5, center=True, scale=True, is_training=True))

		self.W_conv2_3 = self.weight_variable([3,3,32,32],"wc2_3") 
		self.b_conv2_3 = self.bias_variable([32],"bc2_3")
		self.output_conv2_3 = self.conv2d_2x2(self.h_conv2_2,self.W_conv2_3) + self.b_conv2_3 # (360, 330, 32 featureMaps) ==> (180, 165, 32 featureMaps)
		self.h_conv2_3 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(self.output_conv2_3, \
				decay=0.9, updates_collections=None, epsilon=1e-5, center=True, scale=True, is_training=True))
		# CNN layer 3
		self.W_conv3_1 = self.weight_variable([3,3,32,64],"wc3_1") 
		self.b_conv3_1 = self.bias_variable([64],"bc3_1")
		self.output_conv3_1 = self.conv2d(self.h_conv2_3,self.W_conv3_1) + self.b_conv3_1 # (180, 165, 32 featureMaps) ==> (180, 165, 64 featureMaps)
		self.h_conv3_1 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(self.output_conv3_1, \
				decay=0.9, updates_collections=None, epsilon=1e-5, center=True, scale=True, is_training=True))

		self.W_conv3_2 = self.weight_variable([3,3,64,64],"wc3_2") 
		self.b_conv3_2 = self.bias_variable([64],"bc3_2")
		self.output_conv3_2 = self.conv2d(self.h_conv3_1,self.W_conv3_2) + self.b_conv3_2 # (180, 165, 64 featureMaps) ==> (180, 165, 64 featureMaps)
		self.h_conv3_2 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(self.output_conv3_2, \
				decay=0.9, updates_collections=None, epsilon=1e-5, center=True, scale=True, is_training=True))

		self.W_conv3_3 = self.weight_variable([3,3,64,64],"wc3_3") 
		self.b_conv3_3 = self.bias_variable([64],"bc3_3")
		self.output_conv3_3 = self.conv2d_2x2(self.h_conv3_2,self.W_conv3_3) + self.b_conv3_3 # (180, 165, 64 featureMaps) ==> (90, 83, 64 featureMaps)
		self.h_conv3_3 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(self.output_conv3_3, \
				decay=0.9, updates_collections=None, epsilon=1e-5, center=True, scale=True, is_training=True))
		# my net layer 4
		self.W_conv4_1 = self.weight_variable([3,3,32,64],"wc4_1")
		self.output_conv4_1 = self.deconv2d(self.h_conv3_3,self.W_conv4_1, [180, 165], 32) # (90, 83, 64 featureMaps) ==> (180, 165, 32 featureMaps)
		self.h_conv4_1 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(self.output_conv4_1, \
				decay=0.9, updates_collections=None, epsilon=1e-5, center=True, scale=True, is_training=True))

		self.W_conv4_2 = self.weight_variable([3,3,16,32],"wc4_2")
		self.output_conv4_2 = self.deconv2d(self.h_conv4_1,self.W_conv4_2, [360, 330], 16) # (180, 165, 32 featureMaps) ==> (360, 330, 16 featureMaps)
		self.h_conv4_2 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(self.output_conv4_2, \
				decay=0.9, updates_collections=None, epsilon=1e-5, center=True, scale=True, is_training=True))

		self.W_conv4_3 = self.weight_variable([3,3,1,16],"wc4_3")
		self.h_conv4_3 = tf.nn.sigmoid(self.deconv2d(self.h_conv4_2,self.W_conv4_3, [720, 660], 1)) # (360, 330, 16 featureMaps) ==> (720, 660, 1 featureMaps)
		self.prediction = tf.reshape(self.h_conv4_3, [-1,720,660])

		# self.cross_entropy = tf.reduce_mean(tf.log(tf.square(self.ys - self.prediction*0.99998+0.00001)))
		self.cross_entropy = tf.reduce_mean(tf.square(self.ys - self.prediction))
		self.train_step = tf.train.AdamOptimizer(self.learningRate).minimize(self.cross_entropy)

	def train(self, iter, isInit=True):
		self.saver = tf.train.Saver()
		self.sess = tf.Session()
		if isInit:
			self.sess.run(tf.global_variables_initializer()) # important step
		else:
			self.saver.restore(self.sess, "Model/model") 
			print("pre_train Model Restored!")
		# Training 
		data = Data(batchSize=self.batchSize)
		for i in range(iter):
			imageBatch, gtBatch = data.get_next_batch_train_data()
			self.sess.run(self.train_step, feed_dict={self.images:imageBatch, self.ys:gtBatch})
			if i % 100 == 0:
				loss = self.sess.run(self.cross_entropy, feed_dict={self.images:imageBatch, self.ys:gtBatch})
				print('loss :', loss)
				saver_path = self.saver.save(self.sess, "Model/model", global_step=i) 
				print("Model saved in file:", saver_path)
		loss = self.sess.run(self.cross_entropy, feed_dict={self.images:imageBatch, self.ys:gtBatch})
		print('Final loss :', loss)
		saver_path = self.saver.save(self.sess, "Model/model") 
		print("Model saved in file:", saver_path)

	def load_model(self):
		self.saver = tf.train.Saver()
		self.sess = tf.Session()
		self.saver.restore(self.sess, 'Model/model') 

	def predict(self, imageNum):
		imageName = 'image/'+ str(imageNum) +'.jpg' 
		image = cv2.imread(imageName)
		testImage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).reshape((1,720,660,1)).astype(np.float32)/127.5-1
		pre = self.sess.run(self.prediction, feed_dict={self.images:testImage}).reshape(720,660)
		binary = np.around(pre).astype('uint8')
		# cv2.imshow("Image", binary*255)   
		# cv2.waitKey(0)
		cv2.imwrite("./result/fcn167.jpg", binary*255)

		# img, contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
		# truePoint = list()
		# for c in contours:
		# 	if c.shape[0] <= 10:
		# 		continue
		# 	M = cv2.moments(c)
		# 	cX = int(M["m10"] / M["m00"])
		# 	cY = int(M["m01"] / M["m00"])
		# 	truePoint.append([cX, cY])
		# return truePoint


	def test(self):
		self.saver = tf.train.Saver()
		self.sess = tf.Session()
		self.saver.restore(self.sess, 'Model/pre_train/pretrain') 

		self.ys = tf.placeholder(tf.float32, [None, 83, 90, 5])
		self.y_truth = tf.reshape(self.ys, [-1,5])
		# the error between prediction and real data
		self.cross_entropy = tf.reduce_sum(tf.reduce_sum(tf.square(self.y_truth - self.prediction), reduction_indices=1))
		# Choose the optimizer and minimize the loss
		self.train_step = tf.train.AdamOptimizer(1e-3).minimize(self.cross_entropy)

		print("pre_train Model Restored!")
		# del self.saver
		# self.sess.run(tf.global_variables_initializer()) # important step
		# Training 
		data = Data(batchSize=self.batchSize)
		imageBatch, gtBatch = data.get_next_batch_train_data()
		a = self.sess.run(self.h_conv4_6, feed_dict={self.images:imageBatch, self.ys:gtBatch})
		print(np.amax(a))
		print(self.sess.run(self.cross_entropy, feed_dict={self.images:imageBatch, self.ys:gtBatch}))


if __name__ == "__main__":
	# net = FCN(batchSize=10)
	# net.build()
	# net.train(2000, isInit=False)
	

	net = FCN(batchSize=1)
	net.build()
	net.load_model()
	print(net.predict(167))


	# net = FCN(batchSize=1)
	# net.build()
	# net.load_model()
	# VehiclePosition = VehiclePosition()
	# start_time = time.time()
	# for i in range(900):
	# 	print('Predicting NO.'+str(i+1))
	# 	net.predict(i+1)
	# 	truePoint = net.predict(i+1)
	# 	if i > 5:
	# 		vehiclePosition = VehiclePosition.get_vehicles(truePoint)
	# 		result_add_frames(i, vehiclePosition)
	# 		VehiclePosition.update_vehicle_buffer(truePoint, delPre=True)
	# 	else:
	# 		result_add_frames(i, truePoint)
	# 		VehiclePosition.update_vehicle_buffer(truePoint)

	# print('It took', str(time.time()-start_time), 's')

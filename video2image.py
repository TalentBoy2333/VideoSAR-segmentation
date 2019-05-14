import cv2

# load image
# https://www.sandia.gov/radar/video/index.html
video = cv2.VideoCapture('eubankgateandtrafficvideosar.mp4') 
c = 1

if video.isOpened(): 
	rval, frame = video.read()
	print('Loading the VideoSAR : OK')
else:
	rval = False
	print('Loading the VideoSAR : FAIL')

timeF = 1 # catch a frame from (timeF) frames

while rval: 
	rval, frame = video.read()
	if c % timeF == 0 and c >= 150: 
		cv2.imwrite('./image/' + str(c-149) + '.jpg',frame)
	c = c + 1
	cv2.waitKey(1)
video.release()
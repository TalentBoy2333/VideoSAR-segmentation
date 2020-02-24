import cv2 
import numpy as np 
import os 
import matplotlib.pyplot as plt 

'''
Config of Eval
'''
frames_path = '../dataset/frames/' # 视频帧路径 

'''=============================================================='''

def optical_flow(dis, frame1, frame2):
    '''
    :param dis: cv2的光流计算模块, dis = cv2.DISOpticalFlow_create()
    :param frame1: 时间靠前的一帧图像(三通道图像)
    :param frame2: 时间靠后的一帧图像(三通道图像)
    :return bgr: 光流图像(三通道图像)
    '''
    # cv2的光流计算必须要输入灰度图
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255
    # 计算光流
    flow = dis.calc(prvs, next, None,)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang * 180 / np.pi / 2
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # plt.figure()
    # plt.imshow(bgr)
    # plt.show()
    return bgr

def cal_2frames_optical_flow(ind1, ind2):
    frame1_name = str(ind1) + '.png'
    frame1_path = os.path.join(frames_path, frame1_name)
    frame2_name = str(ind2) + '.png'
    frame2_path = os.path.join(frames_path, frame2_name)

    frame1 = cv2.imread(frame1_path)
    frame2 = cv2.imread(frame2_path)

    dis = cv2.DISOpticalFlow_create()
    optical_flow_image = optical_flow(dis, frame1, frame2)
    
    optical_flow_name = str(ind1) + '_' + str(ind2) + '.png'
    cv2.imwrite(optical_flow_name, optical_flow_image)

if __name__ == '__main__':
    cal_2frames_optical_flow(142, 150)

    
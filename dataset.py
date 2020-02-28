import torch.utils.data as Data
import torch
import os
import cv2 
import numpy as np 

'''
Config of dataset
'''
frame_num = 900 # 视频帧数量
image_num_in_frame = 5 # 一个样本包含的视频帧数量
frame_h = 720 # 视频帧行数
frame_w = 660 # 视频帧列数
'''=============================================================='''

class VideoSAR(Data.Dataset):
    '''
    VideoSAR dataset
    '''
    def __init__(self, frames_path='./dataset/frames/', 
                 seg_path='./dataset/segmentation/', 
                 image_size=512, # 随机裁取视频帧中的一个区域作为训练样本 
                 crop_random=True # 是否随机裁剪
                 ):
        self.frames_path = frames_path
        self.seg_path = seg_path
        self.image_size = image_size
        self.crop_random = crop_random
        self.frames_inds = [i for i in range(image_num_in_frame, frame_num+1)]
        self.image_number = len(self.frames_inds)
        

    def __getitem__(self, index):
        """
        torch.utils.data.DataLoader can find this function and get data.
        :param index: 视频帧第一张图像的ind
        :return frames: 一个连续视频帧图像训练样本 [frame, channel, h, w]
        :return seg: 一个连续视频帧图像训练样本的GroundTruth [h, w]
        """
        frames, seg = self.pull_item(index)
        return frames, seg

    def __len__(self):
        """
        This function can: 
        dataset = VideoSAR()
        dataset_size = len(dataset)
        """
        return self.image_number

    def pull_frames(self, index, x, y):
        """
        提取连续视频帧图像
        :param index: 视频帧第一张图像的ind
        :param x: 视频帧裁剪部分左上角起点横坐标
        :param y: 视频帧裁剪部分左上角起点纵坐标
        :return: 一个连续视频帧图像训练样本 [frame, channel, h, w]
        """
        frames = None
        image_size = self.image_size
        for i in range(image_num_in_frame):
            frame_ind = index + i
            frame_name = str(frame_ind) + '.png'
            frame_path = os.path.join(self.frames_path, frame_name)
            frame = cv2.imread(frame_path, 0)
            image = frame[y:y+image_size, x:x+image_size]
            image = torch.Tensor(image) # [h,w]
            image = torch.unsqueeze(image, 0) # 扩展到channel维度[c,h,w]
            image = torch.unsqueeze(image, 0) # 扩展到frame维度[f,c,h,w]
            if frames is None:
                frames = image
            else:
                frames = torch.cat((frames, image), 0)
        return frames

    def pull_segmentation(self, index, x, y):
        """
        提取连续视频帧图像最后一张图像的语义分割标注GroundTruth
        :param index: 视频帧第一张图像的ind
        :param x: 视频帧裁剪部分左上角起点横坐标
        :param y: 视频帧裁剪部分左上角起点纵坐标
        :return: 一个连续视频帧图像训练样本的GroundTruth [h, w]
        """
        image_size = self.image_size
        seg_ind = index + image_num_in_frame - 1
        seg_name = str(seg_ind) + '.png'
        seg_path = os.path.join(self.seg_path, seg_name)
        seg = cv2.imread(seg_path, 0)
        label_num = np.amax(seg) # 语义标注中前景的标注数字
        seg = seg[y:y+image_size, x:x+image_size] / label_num # 得到前景为1, 背景为0的GroundTruth
        seg = torch.Tensor(seg) # [h,w]
        return seg

    def pull_item(self, index):
        """
        得到一个训练样本, 包含连续视频帧与GroundTruth
        :param index: 视频帧第一张图像的ind
        :return frames: 一个连续视频帧图像训练样本 [frame, channel, h, w]
        :return seg: 一个连续视频帧图像训练样本的GroundTruth [h, w]
        """
        image_size = self.image_size
        # 由于index从0开始, 但是视频帧是从'1.png'开始的, 因此, 需要index+1
        index = index + 1
        if self.crop_random:
            # 随机裁取视频帧中的一部分图像作为训练样本
            x = np.random.randint(0, frame_w - image_size) # 裁剪部分左上角起点横坐标
            y = np.random.randint(0, frame_h - image_size) # 裁剪部分左上角起点纵坐标
        else:
            x = (frame_w - image_size) // 2
            y = (frame_h - image_size) // 2
        frames = self.pull_frames(index, x, y)
        seg = self.pull_segmentation(index, x, y)
        return frames, seg

if __name__ == '__main__':
    batch_size = 3
    dataset = VideoSAR()
    dataset_size = len(dataset)
    print('dataset_size:', dataset_size)
    epoch_size = dataset_size // batch_size
    data_loader = Data.DataLoader(dataset, batch_size, num_workers=0, shuffle=True, pin_memory=True)
    batch_iterator = iter(data_loader)

    for iteration in range(epoch_size+1):
        frames, seg = batch_iterator.next()
        print('iter', iteration+1)
        print('images:', frames.size())
        print('labels:', seg.size())
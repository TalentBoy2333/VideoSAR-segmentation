import torch
import torch.nn as nn
import numpy as np 

'''
Config of 3D ResNet
'''
input_size = (64, 64) # (height, width)
hid_dim_layer1 = [64, 64, 64] # layer1中各输出隐藏层
hid_dim_layer2 = [64, 64, 64] # layer2中各输出隐藏层
hid_dim_layer3 = [64, 64, 64] # layer3中各输出隐藏层
hid_dim_layer4 = [64, 64, 64] # layer4中各输出隐藏层
frame_num = 5 # 样本帧数
'''=============================================================='''


class Res3DBlock(nn.Module):
    '''
    3D ResNet Block
    '''
    def __init__(self, channel=64):
        super(Res3DBlock, self).__init__() 
        self.bn1 = nn.BatchNorm3d(channel)
        self.conv1 = nn.Conv3d(channel, channel, kernel_size=(3, 3, 3), padding=(2, 2, 2), dilation=2)
        # self.conv1 = nn.Conv3d(channel, channel, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(channel) 
        self.conv2 = nn.Conv3d(channel, channel, kernel_size=(3, 3, 3), padding=(2, 2, 2), dilation=2)
        # self.conv2 = nn.Conv3d(channel, channel, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        self.relu = nn.ReLU() 

    def forward(self, x):
        h = self.bn1(x)
        h = self.relu(h)
        h = self.conv1(h)
        h = self.bn2(h)
        h = self.relu(h)
        h = self.conv2(h)
        output = h + x 
        # output = h
        return output


class CNN3D(nn.Module):
    '''
    3D CNN backbone
    '''
    def __init__(self):
        super(CNN3D, self).__init__()

        # class torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        channels = hid_dim_layer1
        self.layer1 = nn.Sequential(
            nn.Conv3d(1, channels[0], kernel_size=(3, 3, 3), padding=(1, 1, 1)), 
            nn.BatchNorm3d(channels[0]), 
            nn.ReLU(), 
            nn.Conv3d(channels[0], channels[1], kernel_size=(3, 3, 3), padding=(1, 1, 1)), 
            nn.BatchNorm3d(channels[1]), 
            nn.ReLU(),
            nn.Conv3d(channels[1], channels[2], kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)), 
        )

        channels = hid_dim_layer2
        self.layer2 = nn.Sequential(
            Res3DBlock(channels[0]), 
            Res3DBlock(channels[1]), 
            nn.BatchNorm3d(channels[1]), 
            nn.ReLU(),
            nn.Conv3d(channels[1], channels[2], kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)), 
        )
    
        channels = hid_dim_layer3
        self.layer3 = nn.Sequential(
            Res3DBlock(channels[0]), 
            Res3DBlock(channels[1]), 
            nn.BatchNorm3d(channels[1]), 
            nn.ReLU(),
            nn.Conv3d(channels[1], channels[2], kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(0, 1, 1)), 
        )

        channels = hid_dim_layer4
        self.layer4 = nn.Sequential(
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True), 
            nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(inplace=True), 
        )


    def forward(self, x):
        '''
        :param x: 神经网络输入, [batch, channels, frames, height, width]
        :return h: 隐藏层layer1输出
        :return h1: 隐藏层layer2输出
        :return h2: 隐藏层layer3输出
        :return output: 输出层输出
        '''
        h1 = self.layer1(x)
        # print('layer1 output: ', h.size())
        h2 = self.layer2(h1)
        # print('layer2 output: ', h1.size())
        h3 = self.layer3(h2)
        # print('layer3 output: ', h2.size())

        # 将输出[2, 64, 1, 3, 3]转换为[2, 64, 3, 3]
        output = torch.squeeze(h3, 2)
        output = self.layer4(output)

        return h1, h2, output


if __name__ == '__main__':
    cnn3d = CNN3D()
    # print(cnn3d)

    '''
    # create fake data:
    # batch size: 2 
    # channel: 1
    # frame number: 9
    # image size h: 512
    # image size w: 512
    '''
    fake_data = np.random.uniform(size=[2, 1, frame_num, input_size[0], input_size[1]])
    fake_data = fake_data.astype(np.float32)
    fake_data = torch.from_numpy(fake_data)

    print(fake_data.size())
    h1, h2, output = cnn3d(fake_data)
    print(output.size())

'''
运行结果: 
torch.Size([2, 1, 5, 64, 64])
layer1 output:  torch.Size([2, 64, 5, 32, 32])
layer2 output:  torch.Size([2, 64, 3, 16, 16])
layer3 output:  torch.Size([2, 64, 1, 8, 8])
torch.Size([2, 64, 8, 8])

'''
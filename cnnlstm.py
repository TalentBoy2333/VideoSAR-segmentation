import torch 
import torch.nn as nn 
import numpy as np 
from convlstm import ConvLSTM

'''
Config of Conv LSTM
'''
input_size = (128, 128) # (height, width)
input_dim = 64 # channels
lstm_channel = 64 # LSTM中隐藏层通道数
frame_num = 5 # 样本帧数
down_in_dim = 1 # 下采样层 down_sample 中的输入通道数
down_out_dim = 64 # 下采样层 down_sample 中的输出通道数

up_in_dim = 64 # 下采样层 up_sample 中的输入通道数
up_out_dim = 64 # 下采样层 up_sample 中的输出通道数
'''=============================================================='''

class DownSample(nn.Module):
    '''
    Down Sample Layer
    '''
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)), 
            nn.BatchNorm3d(out_channels), 
            nn.ReLU(), 
            nn.Conv3d(out_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)), 
            nn.BatchNorm3d(out_channels), 
            nn.ReLU(), 
            nn.Conv3d(out_channels,out_channels, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)), 
        )

        self.layer2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)), 
            nn.BatchNorm3d(out_channels), 
            nn.ReLU(), 
            nn.Conv3d(out_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)), 
            nn.BatchNorm3d(out_channels), 
            nn.ReLU(), 
            nn.Conv3d(out_channels,out_channels, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)), 
        )

        self.layer3 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)), 
            nn.BatchNorm3d(out_channels), 
            nn.ReLU(), 
            nn.Conv3d(out_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)), 
            nn.BatchNorm3d(out_channels), 
            nn.ReLU(), 
        )

    def forward(self, x):
        output = self.layer1(x)
        output = self.layer2(output)
        output = self.layer3(output)
        # print(output.size())
        # [n,c,f,h,w] -> [n,f,c,h,w] ConvLSTM输入格式
        convlstm_output = output.permute(0, 2, 1, 3, 4)
        return convlstm_output

class UpSample(nn.Module):
    '''
    Up Sample Layer
    '''
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.up_sample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        output = self.up_sample(x)
        output = self.double_conv(output)
        return output

class CNNLSTM(nn.Module):
    '''
    Conv LSTM backbone
    '''
    def __init__(self):
        super(CNNLSTM, self).__init__()

        self.input_size = input_size 
        self.input_dim = input_dim
        self.hidden_dim = [lstm_channel for _ in range(frame_num)]
        self.num_layers = frame_num

        self.down_sample = DownSample(down_in_dim, down_out_dim)

        self.convlstm = ConvLSTM(
            input_size=self.input_size,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            kernel_size=(3, 3),
            num_layers=self.num_layers,
            batch_first=True, 
            bias=True,
            return_all_layers=False
        )

        self.up_sample = UpSample(up_in_dim, up_out_dim)
    
    def forward(self, x):
        '''
        :param x: Conv LSTM输入, [batch, frames, channels, height, width]
        :return f1: 下采样层1输出
        :return f2: 下采样层2输出
        '''
        feature_map = self.down_sample(x)
        layer_output_list, _ = self.convlstm(feature_map)
        # 'layer_output_list[-1]'是最后一次循环后所有LSTM Cell的输出:
        # layer_output_list[-1] 的 size: [2, 9, 64, 64, 64]
        # 我们只需要最后一个LSTM Cell的输出, 因此取[:,-1,:,:,:]
        # 得到的特征图convlstm_output 的 size: [2, 64, 64, 64]
        convlstm_output = layer_output_list[-1][:,-1,:,:,:]

        # print('Conv Lstm output:', convlstm_output.size())

        f1 = convlstm_output
        f2 = self.up_sample(f1)

        return f1, f2

if __name__ == '__main__':
    conv_lstm = CNNLSTM() 
    conv_lstm = conv_lstm.cuda()

    '''
    # create fake data:
    # batch size: 2 
    # frame number: 9
    # channels: 1
    # height: 64
    # width: 64
    '''
    fake_data = np.random.uniform(size=[2, 1, frame_num, 512, 512])
    fake_data = fake_data.astype(np.float32)
    fake_data = torch.from_numpy(fake_data)
    fake_data = fake_data.cuda()

    print(fake_data.size())
    f1, f2 = conv_lstm(fake_data)
    print('up sample layer1:', f1.size())
    print('up sample layer2:', f2.size())

'''
运行结果:
torch.Size([2, 5, 1, 512, 512])
down sample layer1: torch.Size([2, 64, 128, 128])
down sample layer2: torch.Size([2, 64, 256, 256])
'''

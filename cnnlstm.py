import torch 
import torch.nn as nn 
import numpy as np 
from convlstm import ConvLSTM

'''
Config of Conv LSTM
'''
input_size = (64, 64) # (height, width)
input_dim = 1 # channels
lstm_channel = 64 # LSTM中隐藏层通道数
frame_num = 9 # 样本帧数
down_in_dim1 = 64 # 下采样层 down_sample1 中的输入通道数
down_out_dim1 = 64 # 下采样层 down_sample1 中的输出通道数
down_in_dim2 = 64 # 下采样层 down_sample2 中的输入通道数
down_out_dim2 = 64 # 下采样层 down_sample2 中的输出通道数
down_in_dim3 = 64 # 下采样层 down_sample3 中的输入通道数
down_out_dim3 = 64 # 下采样层 down_sample3 中的输出通道数
'''=============================================================='''

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

        self.down_sample1 = self.down_sample_block(down_in_dim1, down_out_dim1)
        self.down_sample2 = self.down_sample_block(down_in_dim2, down_out_dim2)
        self.down_sample3 = self.down_sample_block(down_in_dim3, down_out_dim3)

    def down_sample_block(self, in_channels, out_channels):
        '''
        对Conv LSTM的输出进行下采样的卷积层模块
        :param in_channels: int
        :param out_channels: int 
        :return layers: 下采样block
        '''
        # class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        layers = torch.nn.Sequential(
            nn.BatchNorm2d(in_channels), 
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(in_channels), 
            nn.ReLU(), 
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(in_channels), 
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1), 
        )
        return layers
    
    def forward(self, x):
        '''
        :param x: Conv LSTM输入, [batch, frames, channels, height, width]
        :return f1: 下采样层1输出
        :return f2: 下采样层2输出
        :return f3: 下采样层3输出
        '''
        layer_output_list, last_state_list = self.convlstm(x)
        # 'layer_output_list[-1]'是最后一次循环后所有LSTM Cell的输出:
        # layer_output_list[-1] 的 size: [2, 9, 64, 64, 64]
        # 我们只需要最后一个LSTM Cell的输出, 因此取[:,-1,:,:,:]
        # 得到的特征图convlstm_output 的 size: [2, 64, 64, 64]
        convlstm_output = layer_output_list[-1][:,-1,:,:,:]

        # print('Conv Lstm output:', convlstm_output.size())

        f1 = self.down_sample1(convlstm_output)
        f2 = self.down_sample2(f1)
        f3 = self.down_sample3(f2)

        return f1, f2, f3

if __name__ == '__main__':
    conv_lstm = CNNLSTM() 

    '''
    # create fake data:
    # batch size: 2 
    # frame number: 9
    # channels: 1
    # height: 64
    # width: 64
    '''
    fake_data = np.random.uniform(size=[2, frame_num, 1, input_size[0], input_size[1]])
    fake_data = fake_data.astype(np.float32)
    fake_data = torch.from_numpy(fake_data)

    print(fake_data.size())
    f1, f2, f3 = conv_lstm(fake_data)
    print('down sample layer1:', f1.size())
    print('down sample layer2:', f2.size())
    print('down sample layer3:', f3.size())

'''
运行结果:
torch.Size([2, 9, 1, 64, 64])
Conv Lstm output: torch.Size([2, 64, 64, 64])
down sample layer1: torch.Size([2, 64, 32, 32])
down sample layer2: torch.Size([2, 64, 16, 16])
down sample layer3: torch.Size([2, 64, 8, 8])
'''

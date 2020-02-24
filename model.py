import torch 
import torch.nn as nn 
import numpy as np 
from backbone import CNN3D 
from cnnlstm import CNNLSTM 

'''
Config of Main Model
'''
h1_dim = 64 # 3D CNN输出的特征h1的通道数
h2_dim = 64 # 3D CNN输出的特征h2的通道数
h3_dim = 64 # 3D CNN输出的特征h3的通道数

# 特征: 转置卷机特征 | 3D卷积特征 | LSTM特征 | 光流特征
#         64      +     64     +    64   +    64  = 256
#         64      +     64     +    64            = 192
#         64      +     64                        = 128
up_in_dim1 = 64 # 上采样层up_sample1输入特征通道数    64
up_out_dim1 = 64 # 上采样层up_sample1输出特征通道数 
up_in_dim2 = 64 # 上采样层up_sample2输入特征通道数   64+64+64 = 192
up_out_dim2 = 64 # 上采样层up_sample2输出特征通道数 
up_in_dim3 = 64 # 上采样层up_sample3输入特征通道数   64+64+64 = 192
up_out_dim3 = 64 # 上采样层up_sample3输出特征通道数  
up_in_dim4 = 64 # 上采样层up_sample4输入特征通道数   64+64+64 = 192
up_out_dim4 = 64 # 上采样层up_sample4输出特征通道数 
'''=============================================================='''

class DoubleConv(nn.Module):
    '''
    Double Conv Layers
    '''
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Model(nn.Module):
    '''
    Main Model
    '''
    def __init__(self, frame_first=True, is_add_lstm=False, is_add_opticalFlow=False):
        super(Model, self).__init__()

        self.frame_first = frame_first
        self.add_lstm = is_add_lstm 
        self.add_opticalFlow = is_add_opticalFlow 

        self.cnn3d = CNN3D() 
        self.convlstm = CNNLSTM() 

        self.layer3d_2d_h = nn.Sequential(
            nn.Conv3d(h1_dim, h1_dim, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)),
            nn.Conv3d(h1_dim, h1_dim, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)),
            nn.Conv3d(h1_dim, h1_dim, kernel_size=(3, 3, 3), padding=(0, 1, 1)),
        )
        self.layer3d_2d_h1 = nn.Sequential(
            nn.Conv3d(h2_dim, h2_dim, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)),
            nn.Conv3d(h2_dim, h2_dim, kernel_size=(3, 3, 3), padding=(0, 1, 1)), 
        )
        self.layer3d_2d_h2 = nn.Sequential(
            nn.Conv3d(h3_dim, h3_dim, kernel_size=(3, 3, 3), padding=(0, 1, 1)), 
        )

        # class torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True)
        self.up_sample1 = nn.ConvTranspose2d(up_in_dim1, up_in_dim1, kernel_size=2, stride=2) 
        self.conv1 = DoubleConv(up_in_dim1 * 2, up_out_dim1)

        self.up_sample2 = nn.ConvTranspose2d(up_in_dim2, up_in_dim2, kernel_size=2, stride=2) 
        self.conv2 = DoubleConv(up_in_dim2 * 2, up_out_dim2)

        self.up_sample3 = nn.ConvTranspose2d(up_in_dim3, up_in_dim3, kernel_size=2, stride=2) 
        self.conv3 = DoubleConv(up_in_dim3 * 2, up_out_dim3)

        self.up_sample4 = nn.ConvTranspose2d(up_in_dim4, up_in_dim4, kernel_size=2, stride=2)   
        self.conv4 = DoubleConv(up_in_dim4, up_out_dim4)

        self.out_conv = nn.Conv2d(up_out_dim4, 1, kernel_size=1) 

    def forward(self, x):
        if self.frame_first:
            cnn3d_x = x.permute(0, 2, 1, 3, 4)
            convlstm_x = x
        else:
            cnn3d_x = x 
            convlstm_x = x.permute(0, 2, 1, 3, 4)

        h, h1, h2, output = self.cnn3d(cnn3d_x)
        h_2d = torch.squeeze(self.layer3d_2d_h(h), 2)
        h1_2d = torch.squeeze(self.layer3d_2d_h1(h1), 2)
        h2_2d = torch.squeeze(self.layer3d_2d_h2(h2), 2)
        # print('h_2d:', h_2d.size())
        # print('h1_2d:', h1_2d.size())
        # print('h2_2d:', h2_2d.size())

        if self.add_lstm == False and self.add_opticalFlow == False:
            up1 = self.up_sample1(output)
            print('up1:', up1.size())
            up1 = torch.cat([up1, h2_2d], dim=1)
            up1 = self.conv1(up1)
            print('up1:', up1.size())
            up2 = self.up_sample2(up1)
            print('up2:', up2.size())
            up2 = torch.cat([up2, h1_2d], dim=1)
            up2 = self.conv1(up2)
            print('up2:', up2.size())
            up3 = self.up_sample3(up2)
            print('up3:', up3.size())
            up3 = torch.cat([up3, h_2d], dim=1)
            up3 = self.conv1(up3)
            print('up3:', up3.size())
        
        if self.add_lstm == True and self.add_opticalFlow == False:
            f1, f2, f3 = self.convlstm(convlstm_x) 

            up1 = self.up_sample1(output)
            # print('up1:', up1.size())
            up1 = torch.cat([up1, h2_2d, f3], dim=1)
            # print('up1:', up1.size())
            up2 = self.up_sample2(up1)
            # print('up2:', up2.size())
            up2 = torch.cat([up2, h1_2d, f2], dim=1)
            # print('up2:', up2.size())
            up3 = self.up_sample3(up2)
            # print('up3:', up3.size())
            up3 = torch.cat([up3, h_2d, f1], dim=1)
            # print('up3:', up3.size())

        # TODO
        # if self.add_lstm == True and self.add_opticalFlow == True:
            

        up4 = self.up_sample4(up3)
        up4 = self.conv4(up4)
        print('up4:', up4.size())
        logits = self.out_conv(up4)
        return logits


if __name__ == '__main__':
    model = Model(frame_first=True, is_add_lstm=False, is_add_opticalFlow=False)
    '''
    # create fake data:
    # batch size: 2 
    # channel: 1
    # frame number: 9
    # image size h: 512
    # image size w: 512
    '''
    fake_data = np.random.uniform(size=[2, 9, 1, 512, 512])
    fake_data = fake_data.astype(np.float32)
    fake_data = torch.from_numpy(fake_data)

    print(fake_data.size())
    logits = model(fake_data)
    print('logits:', logits.size())

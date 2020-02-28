import torch 
import torch.nn as nn
import torch.utils.data as Data
import numpy as np 
from dataset import VideoSAR
from model import Model
import os 

cuda = True if torch.cuda.is_available() else False
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

def weights_init(model):
    """
    设计初始化函数
    使用方法: 在创建model后使用PyTorch模型自带的apply()方法进行参数初始化. 
    栗子: model.apply(weights_init)
    :param model: 神经网络模型 torch.nn.Module
    """
    classname = model.__class__.__name__
    if classname.find('Conv3d') != -1: # 这里的Conv和BatchNnorm是torc.nn里的形式
        n = model.kernel_size[0] * model.kernel_size[1] * model.kernel_size[2] * model.out_channels
        model.weight.data.normal_(0, np.sqrt(2. / n))
        if model.bias is not None:
            model.bias.data.zero_()
    elif classname.find('Conv2d') != -1: 
        n = model.kernel_size[0] * model.kernel_size[1] * model.out_channels
        model.weight.data.normal_(0, np.sqrt(2. / n))
        if model.bias is not None:
            model.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        model.weight.data.normal_(1.0, 0.02) # bn层里初始化γ, 服从（1, 0.02）的正态分布
        model.bias.data.fill_(0)  # bn层里初始化β, 默认为0
    elif classname.find('Linear') != -1:
        n = model.weight.size(1)
        model.weight.data.normal_(0, 0.01)
        model.bias.data = torch.ones(model.bias.data.size())


class SoftDiceLoss(nn.Module):
    """
    Soft Dice Loss计算
    """
    def __init__(self):
        super(SoftDiceLoss, self).__init__()
 
    def forward(self, logits, targets):
        '''
        :param logits: model输出的特征图logits
        :param targets: GroundTruth, 前景为1, 背景为0
        :return: soft dice loss
        '''
        num = targets.size(0)
        smooth = 1e-3 # 平滑因子, 防止分母为0

        probs = torch.sigmoid(logits) # 使用sigmoid将logits映射到[0,1]
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)
 
        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        dice_loss = 1 - score.sum() / num
        return dice_loss

def train(batch_size=4, train_epoch=20):
    """
    训练
    :param batch_size: batch size.
    :param train_epoch: training epoch size
    """
    print('Train.')
    print('Use GPU:', cuda)
    print('batch size:', batch_size)
    print('train epoch:', train_epoch)
    dataset = VideoSAR()
    dataset_size = len(dataset)
    print('frames number:', dataset_size)
    epoch_size = dataset_size // batch_size
    print('iteration number in an epoch:', epoch_size)
    print('Loading Dataset..')
    data_loader = Data.DataLoader(dataset, batch_size, num_workers=8, shuffle=True, pin_memory=True)

    print('Building Model..')
    model = Model(is_add_lstm=True, is_add_opticalFlow=False)
    loss_func = SoftDiceLoss()
    if cuda:
        model = torch.nn.DataParallel(model) # GPU并行训练
        model = model.cuda()
        loss_func = loss_func.cuda()
    model.apply(weights_init)
    # 一共训练20个epoch, 前10个epoch学习率为1e-3, 后10个epoch学习率为1e-4
    learning_rate = 1e-3
    # class torch.optim.SGD(params, lr=, momentum=0, dampening=0, weight_decay=0, nesterov=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    
    print('Training..')
    loss_save = []
    for epoch in range(train_epoch):
        # 一共训练20个epoch, 前10个epoch学习率为1e-3, 后10个epoch学习率为1e-4
        if epoch == 10:
            learning_rate = 1e-4
            for p in optimizer.param_groups:
                p['lr'] = learning_rate
            
        
        batch_iterator = iter(data_loader)
        for iteration in range(1, epoch_size+1):
            frames, seg = batch_iterator.next()
            if cuda:
                frames = frames.cuda() 
                seg = seg.cuda() 
            
            logits = model(frames)
            dice_loss = loss_func(logits, seg)
            optimizer.zero_grad()
            dice_loss.backward()
            optimizer.step()

            loss = dice_loss.cpu().data.numpy()
            loss_save.append(loss)
            print(
                ' | ', 'epoch:', epoch+1, 
                ' | ', 'iter:', iteration, 
                ' | ', 'loss:', loss, 
                ' | ', 'lr:', learning_rate, 
                ' | ', 
            )
        print('Saving parameters in model on epoch', epoch+1)
        torch.save(model.state_dict(), './param/model_epoch'+str(epoch+1).zfill(2)+'.pkl')    
    print('Saving losses during training.')
    np.save('./picture/losses_training.npy', loss_save)       
        
if __name__ == '__main__':
    train()
    # with torch.cuda.device(1):
    #     train()
        

    
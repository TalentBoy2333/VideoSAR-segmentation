import torch 
import torch.nn as nn
import numpy as np 
from dataset import VideoSAR
from model import Model

'''
Config of Eval
'''
image_num_in_frame = 9 # 一个样本包含的视频帧数量
image_size = 512 # 裁取视频帧中的一个区域作为训练样本
frame_h = 720 # 视频帧行数
frame_w = 660 # 视频帧列数
'''=============================================================='''

class CalScore(nn.Module):
    """
    IOU得分计算
    """
    def __init__(self):
        super(CalScore, self).__init__()
 
    def forward(self, logits, targets):
        '''
        :param logits: model输出的特征图logits
        :param targets: GroundTruth, 前景为1, 背景为0
        :return: IOU得分
        '''
        num = targets.size(0)
        smooth = 1e-3 # 平滑因子, 防止分母为0

        probs = torch.sigmoid(logits) # 使用sigmoid将logits映射到[0,1]
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)
 
        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        final_score = score.sum() / num
        return final_score

class CalIOU(nn.Module):
    """
    计算语义分割结果与GroundTruth的IOU
    ref: https://www.aiuai.cn/aifarm1159.html
         https://github.com/pytorch/pytorch/issues/1249
    """
    def __init__(self, threshold=0.5):
        """
        :param threshold: 阈值, 将概率大于threshold的像素点预测为前景.
        """
        super(CalIOU, self).__init__()
        self.threshold = threshold

    def forward(self, logits, targets):
        '''
        :param logits: model输出的特征图logits
        :param targets: GroundTruth, 前景为1, 背景为0
        :return: mask与GroundTruth的IOU
        '''
        num = targets.size(0)
        smooth = 1e-3 # 平滑因子, 防止分母为0

        preds = torch.sigmoid(logits) # 使用sigmoid将logits映射到[0,1]
        th = self.threshold
        preds[preds>=th] = 1. # 概率大于等于threshold的像素点预测为前景, 1
        preds[preds<th] = 0. # 概率小于threshold的像素点预测为背景, 0

        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        iou = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        iou = iou.sum()
    
        return iou, preds


def eval(param_path='./param/model_epoch20.pkl', threshold=0.5):
    """
    Eval. 
    :param param_path: 模型参数路径
    :param threshold: 阈值, 将概率大于threshold的像素点预测为前景.
    """
    print('Eval.')
    # 固定裁剪区域
    dataset = VideoSAR(crop_random=False)
    dataset_size = len(dataset)
    print('frames number:', dataset_size)
    data_loader = Data.DataLoader(dataset, 1, num_workers=8, pin_memory=True)
    batch_iterator = iter(data_loader)

    model = Model()
    model.eval()
    cal_score = CalScore() 
    cal_iou = CalIOU()
    if cuda:
        model = model.cuda()
        cal_score = cal_score.cuda() 
        cal_iou = cal_iou.cuda(threshold) 
    print('Loading parameters of model.')
    model.load_state_dict(torch.load(param_path))

    ious = [] 
    scores = []
    for i in range(dataset_size): 
        frames, seg = batch_iterator.next()
        if cuda:
            frames = frames.cuda() 
            seg = seg.cuda() 
        logits = model(frames)

        score = cal_score(logits, seg)
        iou, _ = cal_iou(logits, seg)
        score = score.cpu().data.numpy()
        iou = iou.cpu().data.numpy() 
        ious.append(iou)
        scores.append(score) 
    print('final iou:', np.mean(ious))
    print('final score:', np.mean(scores))

    print('Saving evaluation result.')
    ious_name = 'iou_' + param_path[-11:-4] + '_th' + str(threshold) + '.npy'
    score_name = 'score_' + param_path[-11:-4] + '.npy'
    np.save(ious_name, ious)
    np.save(score_name, scores)

def mask_one(index, threshold=0.5, frames_path='./dataset/frames/', segs_path='./dataset/segmentation/'):
    """
    预测一个连续视频帧的语义分割结果
    :param index: 连续视频帧ind
    :param threshold: 阈值, 将概率大于threshold的像素点预测为前景.
    :param frames_path: 视频帧路径
    :param segs_path: 语义分割标注路径
    """
    if index < 1 or index > 892:
        print('index Error.')
        return 
    
    model = Model()
    model.eval()
    cal_score = CalScore() 
    cal_iou = CalIOU()
    if cuda:
        model = model.cuda()
        cal_score = cal_score.cuda() 
        cal_iou = cal_iou.cuda(threshold) 
    print('Loading parameters of model.')
    model.load_state_dict(torch.load(param_path))

    '''
    frame共有四个区域: 左上, 右上, 左下, 右下. 将其分别命名为区域1, 2, 3, 4
    |-------|
    | 1 | 2 |
    |-------|
    | 3 | 4 |
    |-------|
    '''

    '''
    左上
    '''
    # load frames 
    frames = None
    for i in range(image_num_in_frame):
        frame_ind = index + i
        frame_name = str(frame_ind) + '.png'
        frame_path = os.path.join(frames_path, frame_name)
        frame = cv2.imread(frame_path, 0)
        image = frame[0:image_size, 0:image_size]
        image = torch.Tensor(image) # [h,w]
        image = torch.unsqueeze(image, 0) # 扩展到channel维度[c,h,w]
        image = torch.unsqueeze(image, 0) # 扩展到frame维度[f,c,h,w]
        if frames is None:
            frames = image
        else:
            frames = torch.cat((frames, image), 0)
    # load segmentation 
    seg_ind = index + image_num_in_frame - 1
    seg_name = str(seg_ind) + '.png'
    seg_path = os.path.join(segs_path, seg_name)
    seg = cv2.imread(seg_path, 0)
    label_num = np.amax(seg) # 语义标注中前景的标注数字
    seg = seg[0:image_size, 0:image_size] / label_num # 得到前景为1, 背景为0的GroundTruth
    seg = torch.Tensor(seg) # [h,w]
    # predict and evaluation
    logits = model(frames)
    score = cal_score(logits, seg)
    iou, preds = cal_iou(logits, seg)
    score = score.cpu().data.numpy()
    iou = iou.cpu().data.numpy() 
    mask1 = preds.cpu().data.numpy() 
    print('Region 1 score:', score)
    print('Region 1 iou:', iou)

    '''
    右上
    '''
    # load frames 
    frames = None
    for i in range(image_num_in_frame):
        frame_ind = index + i
        frame_name = str(frame_ind) + '.png'
        frame_path = os.path.join(frames_path, frame_name)
        frame = cv2.imread(frame_path, 0)
        image = frame[0:image_size, frame_w-image_size+1:frame_w+1]
        image = torch.Tensor(image) # [h,w]
        image = torch.unsqueeze(image, 0) # 扩展到channel维度[c,h,w]
        image = torch.unsqueeze(image, 0) # 扩展到frame维度[f,c,h,w]
        if frames is None:
            frames = image
        else:
            frames = torch.cat((frames, image), 0)
    # load segmentation 
    seg_ind = index + image_num_in_frame - 1
    seg_name = str(seg_ind) + '.png'
    seg_path = os.path.join(segs_path, seg_name)
    seg = cv2.imread(seg_path, 0)
    label_num = np.amax(seg) # 语义标注中前景的标注数字
    seg = seg[0:image_size, frame_w-image_size+1:frame_w+1] / label_num # 得到前景为1, 背景为0的GroundTruth
    seg = torch.Tensor(seg) # [h,w]
    # predict and evaluation
    logits = model(frames)
    score = cal_score(logits, seg)
    iou, preds = cal_iou(logits, seg)
    score = score.cpu().data.numpy()
    iou = iou.cpu().data.numpy() 
    mask2 = preds.cpu().data.numpy() 
    print('Region 2 score:', score)
    print('Region 2 iou:', iou)

    '''
    左下
    '''
    # load frames 
    frames = None
    for i in range(image_num_in_frame):
        frame_ind = index + i
        frame_name = str(frame_ind) + '.png'
        frame_path = os.path.join(frames_path, frame_name)
        frame = cv2.imread(frame_path, 0)
        image = frame[frame_h-image_size+1:frame_h+1, 0:image_size]
        image = torch.Tensor(image) # [h,w]
        image = torch.unsqueeze(image, 0) # 扩展到channel维度[c,h,w]
        image = torch.unsqueeze(image, 0) # 扩展到frame维度[f,c,h,w]
        if frames is None:
            frames = image
        else:
            frames = torch.cat((frames, image), 0)
    # load segmentation 
    seg_ind = index + image_num_in_frame - 1
    seg_name = str(seg_ind) + '.png'
    seg_path = os.path.join(segs_path, seg_name)
    seg = cv2.imread(seg_path, 0)
    label_num = np.amax(seg) # 语义标注中前景的标注数字
    seg = seg[frame_h-image_size+1:frame_h+1, 0:image_size] / label_num # 得到前景为1, 背景为0的GroundTruth
    seg = torch.Tensor(seg) # [h,w]
    # predict and evaluation
    logits = model(frames)
    score = cal_score(logits, seg)
    iou, preds = cal_iou(logits, seg)
    score = score.cpu().data.numpy()
    iou = iou.cpu().data.numpy() 
    mask3 = preds.cpu().data.numpy() 
    print('Region 3 score:', score)
    print('Region 3 iou:', iou)

    '''
    右下
    '''
    # load frames 
    frames = None
    for i in range(image_num_in_frame):
        frame_ind = index + i
        frame_name = str(frame_ind) + '.png'
        frame_path = os.path.join(frames_path, frame_name)
        frame = cv2.imread(frame_path, 0)
        image = frame[frame_h-image_size+1:frame_h+1, frame_w-image_size+1:frame_w+1]
        image = torch.Tensor(image) # [h,w]
        image = torch.unsqueeze(image, 0) # 扩展到channel维度[c,h,w]
        image = torch.unsqueeze(image, 0) # 扩展到frame维度[f,c,h,w]
        if frames is None:
            frames = image
        else:
            frames = torch.cat((frames, image), 0)
    # load segmentation 
    seg_ind = index + image_num_in_frame - 1
    seg_name = str(seg_ind) + '.png'
    seg_path = os.path.join(segs_path, seg_name)
    seg = cv2.imread(seg_path, 0)
    label_num = np.amax(seg) # 语义标注中前景的标注数字
    seg = seg[frame_h-image_size+1:frame_h+1, frame_w-image_size+1:frame_w+1] / label_num # 得到前景为1, 背景为0的GroundTruth
    seg = torch.Tensor(seg) # [h,w]
    # predict and evaluation
    logits = model(frames)
    score = cal_score(logits, seg)
    iou, preds = cal_iou(logits, seg)
    score = score.cpu().data.numpy()
    iou = iou.cpu().data.numpy() 
    mask4 = preds.cpu().data.numpy() 
    print('Region 4 score:', score)
    print('Region 4 iou:', iou)

    '''
    合并4个区域的mask
    '''
    mask = np.zeros((frame_h, frame_w))
    mask[0:image_size, 0:image_size] += mask1
    mask[0:image_size, frame_w-image_size+1:frame_w+1] += mask2
    mask[frame_h-image_size+1:frame_h+1, 0:image_size] += mask3
    mask[frame_h-image_size+1:frame_h+1, frame_w-image_size+1:frame_w+1] += mask4 

    mask[frame_h-image_size+1:image_size, frame_w-image_size+1:image_size] /= 4
    # 重复区域的像素点采用投票的形式决定其为前景还是背景
    # 2, 3, 4票标记为前景(1)
    # 0, 1票标记为背景(0)
    mask[mask>=0.5] = 1. 
    mask[mask<0.5] = 0. 
    mask = mask * 255
    mask = mask.astype(np.uint8)
    pic_name = './picture/ind' + str(index) + '.png'
    cv2.imwrite(pic_name, mask)

    '''
    画图
    '''
    import matplotlib.pyplot as plt
    plt.figure() 

    # 原图
    frame_ind = index + image_num_in_frame - 1
    frame_name = str(frame_ind) + '.png'
    frame_path = os.path.join(frames_path, frame_name)
    frame = cv2.imread(frame_path)
    plt.subplot(1, 3, 1)
    plt.imshow(frame)
    plt.xticks(())
    plt.yticks(())
    plt.title('source')
    
    # 预测结果
    mask_show = np.zeros((frame_h, frame_w, 3))
    mask_show[:,:,0] = mask
    mask_show[:,:,1] = mask
    mask_show[:,:,2] = mask
    plt.subplot(1, 3, 2)
    plt.imshow(mask_show)
    plt.xticks(())
    plt.yticks(())
    plt.title('mask')

    # GroundTruth
    seg_ind = index + image_num_in_frame - 1
    seg_name = str(seg_ind) + '.png'
    seg_path = os.path.join(segs_path, seg_name)
    seg = cv2.imread(seg_path)
    plt.subplot(1, 3, 3)
    plt.imshow(seg)
    plt.xticks(())
    plt.yticks(())
    plt.title('gt')

    plt.show()


if __name__ == '__main__':
    eval(param_path='./param/model_epoch20.pkl', threshold=0.5)

    mask_one(1, threshold=0.5, frames_path='./dataset/frames/', seg_path='./dataset/segmentation/')

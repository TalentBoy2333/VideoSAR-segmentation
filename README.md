这是一个VideoSAR视频车辆语义分割系统</br></br>

需要环境: `opencv-python`, `pytorch==1.4.0`, `numpy` in `python 3.6` 以及`matplotlib` </br></br>

## 数据
`VideoSAR视频`可以再`sandia`官网下载: https://www.sandia.gov/radar/video/index.html</br>
Eubank Gate and Traffic VideoSAR</br>
This is VideoSAR footage of a gate at Kirtland Air Force Base. The video shows vehicle traffic moving through the gate in all motions and directions. The shadows moving along the road are always at the actual physical location of vehicles. As a vehicle stops the reflected energy of the vehicle falls on top of the shadow.  Once the vehicle continues in motion the shadow is again visible.  The lines moving across the screen are Doppler shifts caused by the moving vehicles.</br></br>

`GroundTruth`是我自己使用`labelme`人工标注的. </br>

## 视频处理
`./dataset/`路径下`video2image.py`文件，可以将视频逐帧分解。</br></br>

标注需要使用`labelme`手动标注。

## 模型设置

网络参数在模型文件`backbone.py`，`cnnlstm.py`，`model.py`中设置。程序不复杂，方便二次开发。

## 训练

运行`train.py`文件，可以进行训练。
```Bash
python train.py
```
网络参数存在`./param/`路径下，没有的话可以自己创建一个。

```Python
torch.save(model.state_dict(), './param/model_epoch'+str(epoch+1).zfill(2)+'.pkl') 
```

## 评价

运行`eval.py`文件可以进行模型评价，将评价结果存在`./picture/`路径下

```Bash
python eval.py
```
在`eval.py`的最后可以运行不同的评价。

1.计算模型iou评分

```Python
with torch.no_grad(): 
    eval(param_path='./param/model_epoch20.pkl', threshold=0.5)
```

2.预测一个连续视频帧的语义分割结果，并进行评价。

```Python
with torch.no_grad(): 
    mask_one(142, threshold=0.5, param_path='./param/model_epoch20.pkl', frames_path='./dataset/frames/', segs_path='./dataset/segmentation/')
```


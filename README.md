这是一个VideoSAR车辆检测系统</br></br>

需要环境: `opencv-python`, `tensorflow==1.8.0`, `numpy` in `python 3.6`</br></br>

## 数据
`VideoSAR视频`可以再`sandia`官网下载: https://www.sandia.gov/radar/video/index.html</br>
Eubank Gate and Traffic VideoSAR</br>
This is VideoSAR footage of a gate at Kirtland Air Force Base. The video shows vehicle traffic moving through the gate in all motions and directions. The shadows moving along the road are always at the actual physical location of vehicles. As a vehicle stops the reflected energy of the vehicle falls on top of the shadow.  Once the vehicle continues in motion the shadow is again visible.  The lines moving across the screen are Doppler shifts caused by the moving vehicles.</br></br>

`GroundTruth`是我自己人工标注的. </br>

## 视频处理
运行`video2image.py`文件
```Bash
python video2image.py
```
将视频逐帧分解到`./image/`路径下</br></br>

修改`fcn.py`文件
```Python
# 训练
net = FCN(batchSize=10)
net.build()
net.train(2000, isInit=False)
# 预测
net = FCN(batchSize=1)
net.build()
net.load_model()
print(net.predict(167))
```
运行`fcn.py`文件
```Bash
python fcn.py
```
将检测结果存在`./result/`路径下</br></br>
运行result2video.py文件
```Bash
python result2video.py
```
将检测结果转为视频.


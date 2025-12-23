# CS4277/CS5477: Structure from Motion and Bundle Adjustment

## Setting Up

If you are using Anaconda, you can run the following lines to setup:
```bash
conda create -n sfm python==3.7.6
conda activate sfm
pip  install -r requirements.txt
```

## Running Scripts
To run the scripts:
```bash
python preprocess.py --dataset temple  # performs preprocessing for temple dataset
python preprocess.py --dataset mini-temple  # performs preprocessing for mini-temple dataset
python sfm.py --dataset temple # performs structure from motion without bundle adjustment 
python sfm.py --dataset mini-temple --ba # performs structure from motion with bundle adjustment on mini-temple dataset
python sfm.py --dataset mini-temple # performs structure from motion without bundle adjustment on mini-temple dataset
```

To visualize, run:
```bash
python visualize.py --dataset mini-temple  # visualize 3d point cloud from reconstruction.
```
# 作业要求的pdf在主文件夹内，因为环境问题仅使用终端配置的环境并运行代码，所以在我的pycharm界面中并未配置解释器
### 读取优化后的图像执行这个操作
 python visualize.py --dataset mini-temple --ba 
### 因为优化的数据保存的文件夹不一样
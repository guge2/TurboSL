# TurboSL: 基于神经逆结构光的密集、精确、快速3D重建
### [项目主页](https://www.dgp.toronto.edu/turbosl/) 

## 安装

本项目基于以下源代码构建：
### [基于Instant-NGP的神经表面重建](https://github.com/bennyguo/instant-nsr-pl)

1. 按照上述仓库的步骤安装PyTorch和tiny-cuda-nn。

2. 使用以下命令安装依赖：

```
pip install -r code/requirements.txt
```

## 数据集

`data`目录中提供了结构光图像的示例数据集，包括原始和去畸变图像（`data/images`）、投影图案（`data/patterns`）、标定参数和物体掩码。

## 训练

在`code/configs/neus-sl-geoalbamb-capture-blur.yaml`中配置训练参数，然后运行`code/launch.py`。

训练输出将保存在`code/exp/[experiment_id]`目录下。

参考训练结果见`code/train_david`目录。

## 引用

```
@inproceedings{mirdehghan2024turbosl,
  title={Turbosl: Dense accurate and fast 3d by neural inverse structured light},
  author={Mirdehghan, Parsa and Wu, Maxx and Chen, Wenzheng and Lindell, David B and Kutulakos, Kiriakos N},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={25067--25076},
  year={2024}
}
```

## 致谢

感谢[Yuanchen Guo](https://github.com/bennyguo)提供的Instant-NGP神经表面重建实现。


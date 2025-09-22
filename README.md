# TurboSL: Dense, Accurate and Fast 3D by Neural Inverse Structured Light
### [Project Page](https://www.dgp.toronto.edu/turbosl/) 

## Installation

This project is built on top of the following source code for:
### [Neural Surface Reconstruction based on Instant-NGP](https://github.com/bennyguo/instant-nsr-pl)

1. Follow the steps in the repo above for installing PyTorch and tiny-cuda-nn.

2. Install the requirements file with:

```
pip install -r /code/requirements.txt
```

## Dataset 

A sample dataset for Structured Light images are provided in `/data`. This includes the raw and undistorted images (in `/data/images`), the projection patterns (in `/data/patterns`), the calibration parameters, and object mask.

## Training

To train the model, you can specify the training parameters in `/code/configs/code/configs/neus-sl-geoalbamb-capture-blur.yaml`, and run `/code/launch.py`.

The training outputs will be stored under `/code/exp/[experiment_id]`.

The reference training results are provided under `/code/train_david`.

## Citation

```
@inproceedings{mirdehghan2024turbosl,
  title={Turbosl: Dense accurate and fast 3d by neural inverse structured light},
  author={Mirdehghan, Parsa and Wu, Maxx and Chen, Wenzheng and Lindell, David B and Kutulakos, Kiriakos N},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={25067--25076},
  year={2024}
}
```

## Acknowledgments

We thank [Yuanchen Guo](https://github.com/bennyguo) for their implementation of Neural Surface Reconstruction with Instant-NGP.

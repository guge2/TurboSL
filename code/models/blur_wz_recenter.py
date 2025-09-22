import torch
import torch.nn as nn
import torch.nn.functional as F


import models
from models.utils import get_activation
from models.network_utils import get_encoding, get_mlp, get_encoding_with_network





@models.register('2d-projector-kernel-wz-center')
class BlurKernelWZcenter(nn.Module):
    def __init__(self, config):
        super(BlurKernelWZcenter, self).__init__()

        self.blurstart = config.get('blurstart', -1)
        self.kernel_size = config.kernel_size
        self.halfks = (self.kernel_size - 1) // 2

        self.kernel_x_left = nn.Parameter(torch.zeros(self.halfks,))
        self.kernel_x_right = nn.Parameter(torch.zeros(self.halfks,))

        self.kernel_y_left = nn.Parameter(torch.zeros(self.halfks,))
        self.kernel_y_right = nn.Parameter(torch.zeros(self.halfks,))

        self.kernel_1 = nn.Parameter(torch.ones(1,), requires_grad=False)

        self.normalize = config.get('normalize', 0)
        self.blurprojectorside = config.get('blurprojectorside', 0)
    
    def update_step(self, epoch, global_step):
        self.global_step = global_step

    def savekernel(self,):

        kernel = self.generate_kernel()

        # applying 2D kernel to each channel separately
        kernel = kernel.detach().cpu().numpy()
        kmin = kernel.min()
        kmax = kernel.max()
        kernel = (kernel - kmin) / (kmax - kmin)

        import cv2
        kernel = cv2.resize(kernel, (self.kernel_size * 10, self.kernel_size*10), interpolation = cv2.INTER_NEAREST)
        cv2.imwrite('iter%d-min%.2f-max%.2f.png'%(self.global_step, kmin, kmax), kernel*255)

    def balance_kernel(self, left, right):

        index = torch.arange(1, self.halfks+1).to(left)
        rightmass = right * index
        leftmass = left * (self.halfks + 1 - index)

        # constraint right to match left
        rightblanceratio = leftmass.sum() / (rightmass.sum() + 1e-8)
        right = right * rightblanceratio

        return left, right

    def generate_kernel(self,):

        kx_left, kx_right = self.balance_kernel(self.kernel_x_left, self.kernel_x_right)
        kx_center = 1 - kx_left.sum() - kx_right.sum()
        kx_k = torch.cat([kx_left, kx_center.reshape(1,), kx_right], dim=0)

        ky_left, ky_right = self.balance_kernel(self.kernel_y_left, self.kernel_y_right)
        ky_center = 1 - ky_left.sum() - ky_right.sum()
        ky_k = torch.cat([ky_left, ky_center.reshape(1,),ky_right], dim=0)
        
        kernel_kxk = torch.matmul(kx_k.reshape(-1, 1), ky_k.reshape(1, -1))

        if self.normalize:
            kernel_kxk = kernel_kxk / (kernel_kxk.sum() + 1e-8)

        return kernel_kxk
    
    def apply_blur(self, data_1xkxhxw):
        kernel = self.generate_kernel()
        kernel_1x1xhxw = kernel.unsqueeze(0).unsqueeze(0)

        if False:
            import cv2
            im = x[0,0].detach().cpu().numpy()
            cv2.imshow('im', im)
            cv2.waitKey()

        _, _, h, w = data_1xkxhxw.shape
        assert h == 1024 and w == 1280
        datazero_1xkxhxw = data_1xkxhxw[:, :, 720:]
        assert torch.all(datazero_1xkxhxw==0)

        datapart_1xkxhxw = data_1xkxhxw[:, :, :720]
        datapad_1xkxhxw = F.pad(datapart_1xkxhxw, [self.halfks, self.halfks, self.halfks, self.halfks], mode='constant')
        assert datapad_1xkxhxw.shape[0] == 1
        datapad_kx1xhxw = datapad_1xkxhxw.permute(1, 0, 2, 3)
        output_kx1xhxw = F.conv2d(datapad_kx1xhxw, kernel_1x1xhxw)
        output_1xkxhxw = output_kx1xhxw.permute(1, 0, 2, 3)

        output_1xkxhxw = torch.clip(output_1xkxhxw, 0, 1)
        output_1xkxhxw = torch.cat([output_1xkxhxw, datazero_1xkxhxw], dim=2)

        return output_1xkxhxw


    def forward(self, x):
        output = self.apply_blur(x)

        return output
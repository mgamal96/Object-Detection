
# Anchors/ Priors
# 116,90,  156,198,  373,326 / 32

import torch
import numpy as np

class yoloLayer:
    def __init__(self, img_size):
        self.anchors = torch.Tensor([[ 30.,  61.],[ 62.,  45.], [ 59., 119.]])
        self.imgRows, self.imgCols = img_size
        self.na = 3
        self.no = 85
        self.stride = 16
        self.ny = int(self.imgRows/self.stride)
        self.nx = int(self.imgCols/self.stride)


    def createGrid(self):
        """
        - creates grid of xy coords (1,1,8,13,2) i.e. (bs,1,grid_x,grid_y,2)
        - creates grid of anchor sizes (1,3,8,13,2)
        """

        # build xy offsets
        yv, xv = torch.meshgrid([torch.arange(self.ny), torch.arange(self.nx)])
        self.grid_xy = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2))

        # build Anchors
        self.anchor_wh = self.anchors.view(1,self.na,1,1,2)



    def forward(self, pred):
        """
        Overloaded functionality. Computes box attributes from network logits, given final layer. Also can be
        used to compute box attributes from teacher network output that is already in correct shape, but not calculated.
        i.e. bx = simoid(tx) + x_offset

        Input Shapes allowed - [bs, -1, grid_x , grid y], [bs, 3, -1, grid_x , grid y]
        """
        self.bs = len(pred)
        self.createGrid()

        pred = torch.tensor(pred)

        # Allows for input from NN and logits from teacher network
        if(len(pred.shape) < 5):
            # p.view(bs, 255, 8, 13) -- > (bs, 3, 8, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
            pred = pred.view(self.bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        # Pred - (bs, 3, 8, 13, 85)
        out = pred.clone()
        out[:,:,:,:, :2] = torch.sigmoid(out[:,:,:,:, :2]) + self.grid_xy
        out[:,:,:,:, 2:4] = torch.exp(out[:,:,:,:, 2:4]) * self.anchor_wh
        out[:,:,:,:, 4:] = torch.sigmoid_(out[:,:,:,:, 4:])
        out[:,:,:,:, :2] *= self.stride

        return out


# if __name__ == '__main__':
#
#     pr = torch.randn((1,3,8,13,85))
#     img_size = np.array([256, 416])
#     yl = yoloLayer(img_size)
#     yl.forward(pr)

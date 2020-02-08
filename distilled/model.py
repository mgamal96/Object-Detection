
# Anchors/ Priors
# 116,90,  156,198,  373,326 / 32

import torch
import numpy as np
import yolov3.models as yv3
import torchvision

class yoloLayer:
    """ yolo prediciton layer, applies the expone equations to logits
    """
    def __init__(self, img_size, na=3, nc=4, stride=16):
        self.anchors = torch.Tensor([[ 30.,  61.],[ 62.,  45.], [ 59., 119.]])
        self.imgRows, self.imgCols = img_size
        self.na = na
        self.no = 5 + nc
        self.stride = stride
        self.ny = int(self.imgRows/self.stride)
        self.nx = int(self.imgCols/self.stride)


    def createGrid(self):
        """ Create a grid tensor and anchor size tensor used for computing true x and y by adding offesets.
        Eg.
            creates grid of xy coords, Shape: [1,1,8,13,2] ie. [bs, 1, grid_y, grid_x, 2]
            creates grid of anchor sizes, Shape: [1,3,8,13,2]
        """

        # build xy offsets
        yv, xv = torch.meshgrid([torch.arange(self.ny), torch.arange(self.nx)])
        self.grid_xy = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2))

        # build Anchors
        self.anchor_wh = self.anchors.view(1,self.na,1,1,2)

    def forward(self, pred):
        """ Computes exponentiated output. i.e. true box attributes.  Overloaded functionality.
            1. Computes box attributes from network logits, before reshaping
            2. Compute box attributes from teacher network output that is already reshaped.
        Equations:
                bx = simoid(tx) + x_offset
                bh = px * exp(th)
        Args:
            pred: (tensor) logits, reshaped or not. Shape: [bs, filters, height/stride , width/stride] or
                [bs, 3, filters/3, height/stride, width/stride]
        Returns:
            out: (tensor) exponentiated outputs, Shape: [bs, 3, gridy, gridx, 5+ num classes]
        """
        self.bs = len(pred)
        self.createGrid()

        # Allows logits reshaped for anchors or not
        if(len(pred.shape) < 5):
            # p.view(bs, 255, 8, 13) -- > (bs, 3, 8, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
            pred = pred.view(self.bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        # Pred - (bs, 3, 8, 13, 85)
        # out = pred.clone()
        out = pred
        out[:,:,:,:, :2] = torch.sigmoid(out[:,:,:,:, :2]) + self.grid_xy
        out[:,:,:,:, 2:4] = torch.exp(out[:,:,:,:, 2:4]) * self.anchor_wh
        out[:,:,:,:, 4:] = torch.sigmoid(out[:,:,:,:, 4:])
        out[:,:,:,:, :2] *= self.stride

        return out

class YOLO:
    """ YOLO model. load weights upon initializationa and use to compute model logits, preds and nms preds
    """
    def __init__(self, weightsPath= "yolov3/weights/ultralytics68.pt", modelPath = 'yolov3/cfg/yolov3-spp.cfg'):

        # Create yolo & Initialize YOLO model
        self.weightsPath= weightsPath
        self.model = yv3.Darknet(modelPath, img_size=(256, 416))
        self.model.load_state_dict(torch.load(self.weightsPath)['model'])
        self.model.eval()
        self.nc = 4 # vehicles only will be taken

    def computeLogits(self, data):
        """ Computes logits for VEHICLES ONLY by passing data through YOLOV3 net. Take output from Stride =16
        Args:
            data: (tensor) input data, Shape: [N, 3, height=256, width=416]
        Returns:
            logits_vehicles: (tensor) network output reshaped, Shape [N, Boxes, height/stride, width/stride]
        """

        _, logits = self.model(data)
        bs, _, rows, cols, _ = logits.shape

        logits_vehicles = torch.zeros(bs, 3, rows, cols, self.nc + 5)
        logits_vehicles[..., :2] = logits[..., 2:4]
        logits_vehicles[..., 2] = logits[..., 5]
        logits_vehicles[..., 3] = logits[..., 7]

        return logits_vehicles

    def predict(self, data):
        """ Computes predictions, after passed through exponentiated yolo layer.
        Args:
            data: (tensor) input data, Shape: [N, 3, height=256, width=416]
        Returns:
            preds: (tensor) exponentiated logits, Shape [N, Boxes, height/stride, width/stride]
        """

        bs, _, rows, cols = data.shape
        img_size = np.array([rows, cols])

        logits = self.computeLogits(data)
        yl = yoloLayer(img_size)
        preds = yl.forward(logits)

        return preds #[bs, 3, gridy, gridx, numClasses]

    def non_max_suppression(self, data, iou_thresh=0.5, conf_thresh=0.3):
        """ Computes predictions, after passed through exponentiated yolo layer then performs NMS. Three NMS stages
        1. Threshold to remove low objectness score boxes
        2. Threshold to remove low pr(obj)*pr(class|obj)
        3. Take box with highest confidence when significant IOU overlap b/w boxes

        Args:
            data: (tensor) input data, Shape: [N, 3, height=256, width=416]
            iou_thresh: (float) threshold for (3)
            conf_thresh: (float) threshold for (1) and (2)
        Returns:
            preds: (tensor), Shape [N, 6]
        """
        # Box constraints
        min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
        dets = self.predict(data)

        # Re-shape volume
        N, B, W, H, D = dets.shape
        # preds = dets.reshape(N, B*W*H, D)
        preds = dets.view(N, -1, D)
        output = []


        for k, pred in enumerate(preds): # for image in imgs

            # Min conf
            pred = pred[pred[:, 4] > conf_thresh]

            # Min width and height
            pred = pred[ (pred[:,2:4] > min_wh).all(1) & (pred[:,2:4] < max_wh).all(1) ]


            # Min conf thresh (conf = pr(object)*pr(class|object))
            pred[:, 5:] = pred[:, 5:] * pred[:,4:5]
            box = self.xywh2xyxy(pred[:,:4])
            i, j = (pred[:, 5:] > conf_thresh).nonzero().t()
            pred = torch.cat((box[i], pred[i, j + 5].unsqueeze(1), j.float().unsqueeze(1)), 1)


            c = pred[:, 5]
            output.append(pred[torchvision.ops.boxes.batched_nms(pred[:, :4], pred[:, 4], c, iou_thresh)])

        return output

    def xywh2xyxy(self, x):
        """ Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]. First four in D are xyxy
        Args:
            x: (tensor) old coordiantes, Shape: [N, B*W*H, D]
        Returns:
            y: (tensor) new coords, Shape:[N, B*W*H, D]
        """

        y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:,3] = x[:, 1] + x[:, 3] / 2

        return y

class DISTILLED:
    """ Distlled model. load weights upon initializationa and use to compute model logits, preds and nms preds
    """
    def __init__(self, modelPath ='models/full-model.pt'):
        self.model = torch.load(modelPath)
        self.nc = 4 # num of classes
        self.ac = 3 # num of anchors


    def computeLogits(self, data):
        """ Computes logits by passing data through conv net. Stride =16s
        Args:
            data: (tensor) input data, Shape: [N, 3, height=256, width=416]
        Returns:
            logits: (tensor) network output reshaped, Shape [N, Boxes, height/stride, width/stride]
        """

        logits = self.model(data)
        bs, _, gridy, gridx = logits.shape
        logits = logits.view(bs, self.ac, self.nc +5, gridy, gridx).permute(0, 1, 3, 4, 2).contiguous()

        return logits

    def predict(self, data):
        """ Computes predictions, after passed through exponentiated yolo layer.
        Args:
            data: (tensor) input data, Shape: [N, 3, height=256, width=416]
        Returns:
            preds: (tensor) exponentiated logits, Shape [N, Boxes, height/stride, width/stride]
        """

        # image shape
        bs, _, rows, cols = data.shape
        img_size = np.array([rows, cols])

        logits = self.computeLogits(data)
        yl = yoloLayer(img_size)
        preds = yl.forward(logits)

        return preds

    def non_max_suppression(self, data, iou_thresh=0.5, conf_thresh=0.3):
        """ Computes predictions, after passed through exponentiated yolo layer then performs NMS. Three NMS stages
        1. Threshold to remove low objectness score boxes
        2. Threshold to remove low pr(obj)*pr(class|obj)
        3. Take box with highest confidence when significant IOU overlap b/w boxes

        Args:
            data: (tensor) input data, Shape: [N, 3, height=256, width=416]
            iou_thresh: (float) threshold for (3)
            conf_thresh: (float) threshold for (1) and (2)
        Returns:
            preds: (tensor), Shape [N, 6]
        """

        # Box constraints
        min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
        dets = self.predict(data)


        # Re-shape volume
        N, B, W, H, D = dets.shape
        # preds = dets.reshape(N, B*W*H, D)
        preds = dets.view(N, -1, D)
        output = []



        for k, pred in enumerate(preds): # for image in imgs

            # Min conf
            pred = pred[pred[:, 4] > conf_thresh]

            # Min width and height
            pred = pred[ (pred[:,2:4] > min_wh).all(1) & (pred[:,2:4] < max_wh).all(1) ]


            # Min conf thresh (conf = pr(object)*pr(class|object))
            pred[:, 5:] = pred[:, 5:] * pred[:,4:5]
            box = self.xywh2xyxy(pred[:,:4])
            i, j = (pred[:, 5:] > conf_thresh).nonzero().t()
            pred = torch.cat((box[i], pred[i, j + 5].unsqueeze(1), j.float().unsqueeze(1)), 1)


            c = pred[:, 5]
            output.append(pred[torchvision.ops.boxes.batched_nms(pred[:, :4], pred[:, 4], c, iou_thresh)])

        return output

    def xywh2xyxy(self, x):
        """ Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]. First four in D are xyxy

        Args:
            x: (tensor) old coordiantes, Shape: [N, B*W*H, D]
        Returns:
            y: (tensor) new coords, Shape:[N, B*W*H, D]
        """

        y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:,3] = x[:, 1] + x[:, 3] / 2

        return y

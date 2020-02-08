
import numpy as np
import torch
import torchvision
import cv2
import matplotlib.pyplot as plt
import os
from yolov3 import models as yv3

from model import*


def xywh2xyxy(x):
    """
    Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2] (batch mode)

    Input Shape : [N, B*W*H, D]
    Output Shape : [N, B*W*H, D]
    """

    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:,3] = x[:, 1] + x[:, 3] / 2

    return y

# Full Vectorizing Attempt
def xywh2xyxy_batch(x):
    """
    Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2] (batch mode)

    Input Shape : [N, B*W*H, D]
    Output Shape : [N, B*W*H, D]
    """

    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:,:, 0] = x[:,:, 0] - x[:,:, 2] / 2
    y[:,:, 1] = x[:,:, 1] - x[:,:, 3] / 2
    y[:,:, 2] = x[:,:, 0] + x[:,:, 2] / 2
    y[:,:,3] = x[:,:, 1] + x[:,:, 3] / 2

    return y

# Full Vectorizing Attempt
def non_max_suppression_Vec(dets, iou_thresh, conf_fresh):


    # Box constraints
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height

    # Re-shape volume
    N, B, W, H, D = dets.shape
    preds = dets.reshape(N, B*W*H, D)

    # Width and Height thresholds
    preds[:,:,2:4][preds[:,:,2:4] <= min_wh] = np.zeros_like(preds[:,:,2:4][preds[:,:,2:4] <= min_wh])
    preds[preds[:,:,2:4] >= max_wh] = torch.zeros_like(preds[preds[:,:,2:4] <= max_wh])


    # If none remain process next image
    # if len(preds) == 0:
    #     continue


    # Compute confidence
    # [N, B*W*H, D]   [N, B*W*H, D] * [N, B*W*H, 1]
    pred[:,:, 5:] = pred[:,:, 5:] * pred[:,: 4:5]  # conf = obj_conf * cls_conf

    # Convert coords
    boxes = xywh2xyxy(pred[:,:, :4])
    preds= pred.clone().detach()
    preds[:,:,:4] = boxes

    # Confidence
    i, j = (pred[:, 5:] > conf_thres).nonzero().t()
    pred = torch.cat((box[i], pred[i, j + 5].unsqueeze(1), j.float().unsqueeze(1)), 1)


def non_max_suppression(dets, iou_thresh, conf_thresh):
    """
    Supresses 3 things
    1. Threshold to remove low objectness score boxes
    2. Threshold to remove low pr(obj)*pr(class|obj)
    3. Take box with highest confidence when significant IOU overlap b/w boxes

    Input   - dets: [bs, 3, gridy, gridx, numClasses +5]
    Output  - [[Num detections, 6]* bs]
    """
    # Box constraints
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height

    import pdb; pdb.set_trace()

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
        box = xywh2xyxy(pred[:,:4])
        i, j = (pred[:, 5:] > conf_thresh).nonzero().t()
        pred = torch.cat((box[i], pred[i, j + 5].unsqueeze(1), j.float().unsqueeze(1)), 1)


        c = pred[:, 5]
        output.append(pred[torchvision.ops.boxes.batched_nms(pred[:, :4], pred[:, 4], c, iou_thresh)])

    return output


def plot_predictions(pred, img):
    """
    Plots boinding boxes and labels given the pred [-1, 6], img and scale
    """

    # Scale coordinates

    scale_x = img.shape[1]/416
    scale_y = img.shape[0]/256



    pred[:,0], pred[:,2] = scale_x*pred[:,0], scale_x*pred[:,2]
    pred[:,1], pred[:,3] = scale_y*pred[:,1], scale_y*pred[:,3]



    for i, p in enumerate(pred):
        # import pdb; pdb.set_trace()
        x1, y1, x2, y2, = (np.array(p[:4])).astype(int)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)


    filename = 'savedImage.jpg'
    cv2.imwrite(filename, img)




class IOU:
    def intersect(self, box_a, box_b):
        """ We resize both tensors to [A,B,2] without new malloc:
        [A,2] -> [A,1,2] -> [A,B,2]
        [B,2] -> [1,B,2] -> [A,B,2]
        Then we compute the area of intersect between box_a and box_b.
        Args:
          box_a: (tensor) bounding boxes, Shape: [A,4].
          box_b: (tensor) bounding boxes, Shape: [B,4].
        Return:
          (tensor) intersection area, Shape: [A,B].
        """
        A = box_a.size(0)
        B = box_b.size(0)
        max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                           box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
        min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                           box_b[:, :2].unsqueeze(0).expand(A, B, 2))
        inter = torch.clamp((max_xy - min_xy), min=0)
        return inter[:, :, 0] * inter[:, :, 1]


    def jaccard(self, box_a, box_b):
        """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
        is simply the intersection over union of two boxes.  Here we operate on
        ground truth boxes and default boxes.
        E.g.:
            A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
        Args:
            box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
            box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
        Return:
            jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
        """
        inter = intersect(box_a, box_b)
        area_a = ((box_a[:, 2]-box_a[:, 0]) *
                  (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
        area_b = ((box_b[:, 2]-box_b[:, 0]) *
                  (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
        union = area_a + area_b - inter
        return inter / union  # [A,B]


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, weightsPath="yolov3/weights/ultralytics68.pt", modelPath='yolov3/cfg/yolov3-spp.cfg'):
        self.data_path = data_path
        self.filenames = os.listdir(data_path)

        # Keep only .jpg
        for item in self.filenames:
            if (item[-3:] != 'jpg'):
                self.filenames.remove(item)

        # # Create yolo & Initialize YOLO model
        # self.weightsPath= weightsPath
        # self.model = yv3.Darknet(modelPath, img_size=(256, 416))
        # self.model.load_state_dict(torch.load(self.weightsPath)['model'])
        # self.model.eval()


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.filenames)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        filename = self.filenames[index]

        # Load data and get label
        img = cv2.imread(self.data_path + filename)


        imgrs = cv2.resize(img, (416, 256))
        imgs = np.ascontiguousarray(imgrs, dtype=np.float32)

        X = torch.Tensor(imgs/255.0).permute(2,0,1)
        # Y = self.yv3_predict(X)

        return X

    def yv3_predict(self, imgs):
        """
        Given a batch of images [BS, 3, 256, 416] will return yv3 prediciton [BS, 3, gridx, gridy, 85]
        """

        # Forward passs
        preds, logits = self.model(imgs) #[bs, -1, 85], [bs, 3, gridy, gridx, 85] (before activations)


        return logits


class yolov3_generator:
    """
    Given a path, will generate the output logits for each image in path and save as "yolo-logits-imgname.pt"
    """
    def __init__(self, data_path, labels_path):
        self.data_path = data_path
        self.labels_path = labels_path
        self.generate_labels()

    def yv3_predict(self, imgs):
        """
        Given a batch of images [BS, 3, 256, 416] will return yv3 prediciton [BS, 3, gridx, gridy, 85]
        """

        # Create yolo model

        # Initialize model
        # weightsPath = "yolov3/weights/yolov3-spp.pt"
        weightsPath= "yolov3/weights/ultralytics68.pt"
        model = yv3.Darknet('yolov3/cfg/yolov3-spp.cfg', img_size=(256, 416))

        model.load_state_dict(torch.load(weightsPath)['model'])
        model.eval()

        # Forward pass
        preds, logits = model(imgs) #[bs, -1, 85], [bs, 3, gridy, gridx, 85] (before activations)


        return logits

    def generate_labels(self):
        """
        Given directory of images, passes them through yolov3 and saves the outputs
        """
        filenames = os.listdir(self.data_path)
        for filename in filenames:

            if(filename[0:4] == 'yolo'):
                continue
            elif (filename[-3:] in ["jpg", "png" ]):
                # Pass through yolo
                img = cv2.imread(self.data_path + filename)
                imgrs = cv2.resize(img, (416, 256))
                imgs = np.ascontiguousarray(imgrs, dtype=np.float32)
                imgs = torch.Tensor(imgs/255.0)
                imgs = torch.unsqueeze(imgs, 0)
                imgs = imgs.permute(0,3,1,2)

                # Logits before final exponentiation yolo layer
                logits = self.yv3_predict(imgs)

                # preds after final exponentiation yolo layer
                # yl = yoloLayer(np.array([256, 416]))
                # pred = yl.forward(logits)

                torch.save(imgs, self.labels_path  + "yolo-logits-" + filename[:-4] + ".pt")


import numpy as np
import torch
import torchvision
import cv2
import matplotlib.pyplot as plt

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

    # Box constraints
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height


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
        pred = torch.cat((box[i], pred[i, j + 5].unsqueeze(1), j.double().unsqueeze(1)), 1)


        c = pred[:, 5]
        output.append(pred[torchvision.ops.boxes.batched_nms(pred[:, :4], pred[:, 4], c, iou_thresh)])

    return output


def plot_predictions(pred, img):
    """
    Plots boinding boxes and labels givem the pred [-1, 6], img and scale
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


if __name__ == '__main__':

    dets = np.load("params.npy")
    iou_thresh = 0.5
    conf_fresh = 0.3
    img_size = np.array([256, 416])

    yl = yoloLayer(img_size)
    imgName = "s1.jpg"
    img = cv2.imread(imgName)



    for i, det in enumerate(dets):
        out = yl.forward(dets[i:i+1])
        ot = non_max_suppression(out, iou_thresh, conf_fresh)
        print(ot[0].shape)
        plot_predictions(ot[0], img)
        import pdb; pdb.set_trace()

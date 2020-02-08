import torch
import numpy as np
# import yolov3.models as yv3

import cv2
from model import*
from utils import*

def accuracy(newModel, YOLO,  valid_set_path):
    """
    Compute Precision, Recall, Average Precision

    Args:
        newModel: (DISTILLED CLASS)
        YOLO: (YOLO CLASS)
        valid_set_path: (string) path to validation set
    Returns:
        Precision, Recall, Average Precision
    """

    # Generates data points, normalized and re-sized
    val_set = Dataset(valid_set_path)
    val_generator = torch.utils.data.DataLoader(val_set,
            batch_size =1,
            shuffle =True,
            num_workers =6)

    for x in val_generator:
        gts = yoloModel.non_max_suppression(x)
        preds = newModel.non_max_suppression(x)         # [bs* [N, 9]]

        precision = np.zeros((len(preds)))
        for i, pred in enumerate(preds):                # [N, 9]

            # IOU Adjacency matrix
            IOU = IOU()
            iou = IOU.jaccard(pred[:, :4], gts[i][:,:4])
            iou *= cls_adj # [N1, N2, 1]
            iou = iou[iou > 0.5]

            TPs = iou.sum()
            Total_Ps = len(preds[0])

            precision[i] = TPs/Total_Ps

        mean_precision = precision.mean()


if __name__ == '__main__':

    # Trianing parameters
    learning_rate = 1e-4

    data_path = 'samples/'
    labels_path = 'samples_labels/'


    # Generates data points, normalized and re-sized
    train_set = Dataset(data_path)
    training_generator = torch.utils.data.DataLoader(train_set,
            batch_size =10,
            shuffle =True,
            num_workers =6)

    # Define models, output shape [2, 3, 16, 26, 85]
    yoloModel = YOLO(weightsPath= "yolov3/weights/ultralytics68.pt", modelPath = 'yolov3/cfg/yolov3-spp.cfg')
    newModel = DISTILLED(modelPath ='models/full-model.pt')

    # Loss function and optimizer
    loss_fn = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(newModel.model.parameters(), lr=learning_rate)


    for i in range(100):
        for X in training_generator:

            optimizer.zero_grad()

            y = yoloModel.predict(X).clone().detach()        # torch.Size([2, 3, 16, 26, 9])
            # import pdb; pdb.set_trace()
            pred = newModel.predict(X)      # torch.Size([2, 3, 16, 26, 9])
            loss = loss_fn(pred, y)
            print(loss)
            loss.backward()
            optimizer.step()









    # Logits before final exponentiation yolo layer
    # logits = yv3_predict(imgs)


    # iou_thresh = 0.5
    # conf_fresh = 0.3
    # img_size = np.array([256, 416])
    # yl = yoloLayer(img_size)
    # pred = yl.forward(logits)
    #
    # pred_nms = non_max_suppression(pred, iou_thresh, conf_fresh)
    # plot_predictions(pred_nms[0], (255*np.array(imgs[0].permute(1,2,0))).astype(int) )




# dataPath = 'bdd100k 2/images/10k/train/'
# labelPath = 'bdd100k 2/images/10k/train/'
# teacherLabelPath = 'teacherLabels/'
#
#
# # Obtain teacher label
# imgPath = "bdd100k 2/images/10k/test/ac6d4f42-00000000.jpg"
# img = cv2.imread(imgPath)

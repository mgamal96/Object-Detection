import torch
import numpy as np
import yolov3.models as yv3


def yv3_predict(imgs):
    """
    Given a batch of images [BS, 3, 256, 416] will return yv3 prediciton [BS, 3, gridx, gridy, 85]
    """

    # Create yolo model

    # Initialize model
    model = yv3.Darknet('cfg/yolov3-spp.cfg', img_size=(256, 416))
    model.load_state_dict(torch.load(weights, map_location=device)['model'])
    model.eval()

    # Forward pass
    preds, logits = model(img) #[bs, -1, 85], [bs, 3, gridy, gridx, 85] (before activations)


    return logits


if __name__ == '__main__':


# dataPath = 'bdd100k 2/images/10k/train/'
# labelPath = 'bdd100k 2/images/10k/train/'
# teacherLabelPath = 'teacherLabels/'
#
#
# # Obtain teacher label
# imgPath = "bdd100k 2/images/10k/test/ac6d4f42-00000000.jpg"
# img = cv2.imread(imgPath)

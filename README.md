
# Description

This project uses yolov3 as a teacher network for a smaller network trained only on vehicle objects. 

# Model
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 256, 416]           1,792
              ReLU-2         [-1, 64, 256, 416]               0
            Conv2d-3         [-1, 64, 256, 416]          36,928
              ReLU-4         [-1, 64, 256, 416]               0
         MaxPool2d-5         [-1, 64, 128, 208]               0
            Conv2d-6        [-1, 128, 128, 208]          73,856
              ReLU-7        [-1, 128, 128, 208]               0
            Conv2d-8        [-1, 128, 128, 208]         147,584
              ReLU-9        [-1, 128, 128, 208]               0
        MaxPool2d-10         [-1, 128, 64, 104]               0
           Conv2d-11         [-1, 256, 64, 104]         295,168
             ReLU-12         [-1, 256, 64, 104]               0
           Conv2d-13         [-1, 256, 64, 104]         590,080
             ReLU-14         [-1, 256, 64, 104]               0
           Conv2d-15         [-1, 256, 64, 104]         590,080
             ReLU-16         [-1, 256, 64, 104]               0
        MaxPool2d-17          [-1, 256, 32, 52]               0
           Conv2d-18          [-1, 128, 32, 52]         295,040
             ReLU-19          [-1, 128, 32, 52]               0
           Conv2d-20          [-1, 128, 32, 52]         147,584
             ReLU-21          [-1, 128, 32, 52]               0
        MaxPool2d-22          [-1, 128, 16, 26]               0
           Conv2d-23           [-1, 64, 16, 26]          73,792
             ReLU-24           [-1, 64, 16, 26]               0
           Conv2d-25           [-1, 64, 16, 26]          36,928
             ReLU-26           [-1, 64, 16, 26]               0
           Conv2d-27           [-1, 27, 16, 26]          15,579
             ReLU-28           [-1, 27, 16, 26]               0
           Conv2d-29           [-1, 27, 16, 26]           6,588
================================================================
Total params: 2,310,999
Trainable params: 575,511
Non-trainable params: 1,735,488

# Qualitative Assesment 


<table style="width:100%">
  <tr>
    <th>Our Model<img src="https://github.com/mgamal96/Object-Detection/blob/master/imgs/us3.jpg?raw=true" width="400"></th>
        <th>Yolov3<img src="https://github.com/mgamal96/Object-Detection/blob/master/imgs/yolo3.jpg?raw=true" width="400"></th>
  </tr>
</table>


<table style="width:100%">
  <tr>
    <th>Our Model<img src="https://github.com/mgamal96/Object-Detection/blob/master/imgs/us2.jpg?raw=true" width="400"></th>
        <th>Yolov3<img src="https://github.com/mgamal96/Object-Detection/blob/master/imgs/yolo2.jpg?raw=true" width="400"></th>
  </tr>
</table>

<table style="width:100%">
  <tr>
    <th>Our Model<img src="https://github.com/mgamal96/Object-Detection/blob/master/imgs/us1.jpg?raw=true" width="400"></th>
        <th>Yolov3<img src="https://github.com/mgamal96/Object-Detection/blob/master/imgs/yolo3.jpg?raw=true" width="400"></th>
  </tr>
</table>


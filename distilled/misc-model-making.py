import torchvision.models as models
from torchsummary import summary
import torch


# LOAD VGG-16
PATH = 'models/vgg16-head.pt'
vgg16 = models.vgg16(pretrained=True)
newModel = torch.nn.Sequential(*[vgg16.features[i] for i in range(17)])
model = torch.save(newModel, PATH)

head = torch.load(PATH)

model = torch.nn.Sequential(
    # in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'
    head,
    torch.nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
    torch.nn.ReLU(),
    torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), #(512, 16, 26)

    torch.nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
    torch.nn.ReLU(),
    torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
    torch.nn.ReLU(),
    torch.nn.Conv2d(64, 27, kernel_size=3, stride=1, padding=1),
    torch.nn.ReLU(),
    torch.nn.Conv2d(27, 27, kernel_size=3, stride=1, padding=1)

)

for i, param in enumerate(model.parameters()):
    # print(param_tensor, model.state_dict()[param_tensor].shape, model.state_dict()[param_tensor].requires_grad, )
    param.requires_grad = False
    if(i == 13):
        break

for i, param in enumerate(model.parameters()):
    # print(param_tensor, model.state_dict()[param_tensor].shape, model.state_dict()[param_tensor].requires_grad, )
    print(i, "  ", param.requires_grad , "  ",  param.shape)

summary(model, (3,256, 416))
torch.save(model, 'models/full-model.pt')

import torch.nn as nn
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor  
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

def pretrainedModel(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

class ResidualBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv2 = nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_ch)
        
    def forward(self, x):
        residual = x
        x = nn.ReLU(inplace=True)(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return nn.ReLU(inplace=True)(x + residual)
    
def Model1(num_classes):
    backbone = nn.Sequential(
        nn.Conv2d(3, 64, 3, 2, 1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 128, 3, 2, 1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        ResidualBlock(128),
        nn.Conv2d(128, 256, 3, 1, 1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True)
    )
    backbone.out_channels = 256

    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(['0'], 7, 2)

    return FasterRCNN(backbone, num_classes,
                    rpn_anchor_generator=anchor_generator,
                    box_roi_pool=roi_pooler,
                    min_size=800, max_size=1333)

def Model2(num_classes):
    backbone = nn.Sequential(
        nn.Conv2d(3, 64, 3, 2, 1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        ResidualBlock(64),
        nn.Conv2d(64, 128, 3, 2, 1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        ResidualBlock(128),
        nn.Conv2d(128, 256, 3, 2, 1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.Dropout2d(0.2),
        ResidualBlock(256)
    )
    backbone.out_channels = 256

    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )
    
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(['0'], 7, 2)
    
    return FasterRCNN(backbone, num_classes,
                     rpn_anchor_generator=anchor_generator,
                     box_roi_pool=roi_pooler)
import torch
from torch import nn
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn, faster_rcnn


def create_fasterrcnn(dropout, backbone, num_classes):
    print('Creating Faster-RCNN')
    model = fasterrcnn_resnet50_fpn() if backbone else fasterrcnn_resnet50_fpn(weights_backbone=ResNet50_Weights.IMAGENET1K_V1)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, num_classes)  # Number of dataset classes + 1 (background)

    if dropout:
        # Adding dropout to the 2 fully connected layer
        model.roi_heads.box_head.fc6 = nn.Sequential(
            model.roi_heads.box_head.fc6,
            nn.Dropout(p=0.15))

        model.roi_heads.box_head.fc7 = nn.Sequential(
            model.roi_heads.box_head.fc7,
            nn.Dropout(p=0.15))

    # Freezing backbone
    if not backbone:
        model.backbone.requires_grad_(False)

    model.cuda()

    return model


class our_CNN(nn.Module):
    def __init__(self, num_classes: int = 3, dropout: float = 0.15):
        super().__init__()
        print('Creating our CNN')

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 168, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(168, 392, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(392, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 5 * 5, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

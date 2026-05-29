import torch
from torch import Tensor
import torch.nn as nn

# Yolov1 paper config
architecture_config = [
    (7, 64, 2, 3), # Conv. layer
    "M", # Max pool
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4], # repeate blocks
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),   
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x: Tensor):
        x = self.conv(x)
        x = self.batch_norm(x)
        return self.leaky_relu(x)
    

class YoloV1(nn.Module):
    def __init__(self, in_channels:int = 3, **kwargs) -> None:
        super(YoloV1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x: Tensor):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))
    
    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) is tuple:  # noqa: E721
                layers += [
                    CNNBlock(
                        in_channels,
                        x[1], 
                        kernel_size=x[0], 
                        stride=x[2], 
                        padding=x[3]
                    )
                ]
                in_channels = x[1]

            elif type(x) is str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

            elif type(x) is list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeat = x[2]

                for _ in range(num_repeat):
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3]
                        ),                  
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3]
                        )
                    ]

                    in_channels = conv2[1]
        
        return nn.Sequential(*layers)
    
    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C  = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 496), # Should 4096 from original paper
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + B * 5)) # each box have 5 numbers (x, y, h, w, probability) + num_classes = [cls1, cls2 ... clsn, x, y, w, h, probability]
        )
    
def test(S=7, B=2, C=20):
    model = YoloV1(split_size=S, num_boxes=B, num_classes=C)
    x = torch.randn(2, 3, 448, 448)
    print(model(x).shape)
        
if __name__ == "__main__":
    test()

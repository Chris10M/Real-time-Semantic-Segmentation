import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import geffnet

from .modules import *


class SegNet(nn.Module):
    def __init__(self, n_classes, act_layer=nn.Hardswish):
        super(SegNet, self).__init__()

        backbone = geffnet.create_model('tf_mobilenetv3_small_100', pretrained=True, as_sequential=True)
        self.features = nn.Sequential(*list(backbone)[:-7])

        self.hr_layers = nn.ModuleList([
            InvertedResidualBlock(24, 48, act_layer=act_layer),
            InvertedResidualBlock(48, 48, act_layer=act_layer),
            InvertedResidualBlock(48, 96, act_layer=act_layer),
        ])            

        self.ffm_1 = FeatureFusionModule(40, 48, act_layer=act_layer)
        self.ffm_2 = FeatureFusionModule(96, 48, act_layer=act_layer, stage=2)

        self.pspm = PSPModule(96, 96, act_layer=act_layer)
        self.hr_ffm = HRV2FFM(96, 96, act_layer=act_layer)

        self.seg_head = SegmentationHead(96 + 96, n_classes, act_layer=act_layer, scale=8, expansion=2)
        
    def forward(self, x):
        '''
        2048, 1024   
        1024, 512    / 2
        512, 256     / 4
        256, 128     / 8
        128, 64      / 16
        64, 32       / 32

        '''

        stem = self.features[:3](x)

        stage_1 = self.features[3](stem)
        stage_2 = self.features[4](stage_1)
        
        stage_3 = self.features[5](stage_2)
        hr_stage_3 = self.hr_layers[0](stage_2)

        stage_3, hr_stage_3 = self.ffm_1(stage_3, hr_stage_3)        

        stage_4 = self.features[6:8](stage_3)
        hr_stage_4 = self.hr_layers[1](hr_stage_3)

        stage_4, hr_stage_4 = self.ffm_2(stage_4, hr_stage_4)    

        hr_stage_5 = self.hr_layers[2](hr_stage_4)
        stage_5 = self.pspm(stage_4)

        stage_n = self.hr_ffm(stage_5, hr_stage_5)

        out = self.seg_head(stage_n)

        return out


def get_network(n_classes):
    net = SegNet(n_classes=n_classes)

    if torch.cuda.is_available():
        net = net.cuda()

    net.train()

    return net


def main():
    net = get_network(n_classes=21)
    net.train()

    in_ten = torch.randn(2, 3, 256, 256)

    if torch.cuda.is_available():
        net = net.cuda()
        in_ten = in_ten.cuda()
    
    out = net(in_ten)
    print(out.shape)


if __name__ == "__main__":
    main()
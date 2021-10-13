import torch as torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
import math
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class DSNet(nn.Module):
    """
    Multi-scale semantic feature extraction, contains ResNet50 and Adaptive model.

    Args:
        localScale : first three local feature sizes in Multi-scale feature.
        msfSize:  Multi-scale feature size for full connection layer(fc layer).
    """

    def __init__(self, localScale, msfSize):
        super(DSNet, self).__init__()
        self.msfSize = msfSize

        self.res = resnet50_backbone(localScale, msfSize, pretrained=True)
        self.mapNet = mapGM()
        # initialize
        for i, m_name in enumerate(self._modules):
            if i > 2:
                nn.init.kaiming_normal_(self._modules[m_name].weight.data)

    def forward(self, img, spix):
        spaMap = self.mapNet(spix)
        res_out = self.res(img, spaMap)  # feature fusion
        fe_in_vec = res_out['fe_in_vec'].view(-1, self.msfSize, 1, 1)  # prepare for fc layer
        out = {}
        out['fe_in_vec'] = fe_in_vec

        return out


class mapGM(nn.Module):
    """
    Superpixel adjacency map generation model
    """
    def __init__(self):
        super(mapGM, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(100, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.AdaptiveMaxPool2d((8, 8)),
            nn.Conv2d(32, 16, kernel_size=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.fcLayer = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.Conv2d(512, 48, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.fcLayer2 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, bias=False),
            nn.ReLU(inplace=True)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)

    def forward(self, input):
        midOutput = self.layer(input)
        newOutput = midOutput.view(-1, 1024, 1, 1)
        outputL = self.fcLayer(newOutput)  # for local features in Multi-scale feature
        outputG = self.fcLayer2(newOutput)  # for global features in Multi-scale feature
        output = torch.cat((outputL, outputG), dim=1)
        return output


class PredictNet(nn.Module):
    """
    Prediction network accepts Multi-scale feature to predict final quality score.
    """
    def __init__(self):
        super(PredictNet, self).__init__()
        self.l1 = nn.Sequential(
            nn.Conv2d(224, 112, kernel_size=1),
            nn.LeakyReLU(),
        )
        self.l2 = nn.Sequential(
            nn.Conv2d(112, 14, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(14, 1, kernel_size=1)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        q = self.l1(x)
        q = self.l2(q).squeeze()
        return q


class Bottleneck(nn.Module):
    """
    To build ResNet
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetBackbone(nn.Module):

    def __init__(self, localScale, msfSize, block, layers):
        super(ResNetBackbone, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Adaptive model.
        self.lc1_pool = nn.Sequential(
            nn.Conv2d(256, 16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AdaptiveAvgPool2d((8, 8)),
        )
        self.lc1_fc = nn.Linear(16 * 64, localScale)

        self.lc2_pool = nn.Sequential(
            nn.Conv2d(512, 32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.lc2_fc = nn.Linear(32 * 16, localScale)

        self.lc3_pool = nn.Sequential(
            nn.Conv2d(1024, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.lc3_fc = nn.Linear(64 * 4, localScale)

        self.gl_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.gl_fc = nn.Linear(2048, 512)
        self.gl_fc2 = nn.Linear(512, msfSize - localScale * 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # initialize
        nn.init.kaiming_normal_(self.lc1_pool._modules['0'].weight.data)
        nn.init.kaiming_normal_(self.lc2_pool._modules['0'].weight.data)
        nn.init.kaiming_normal_(self.lc3_pool._modules['0'].weight.data)
        nn.init.kaiming_normal_(self.lc1_fc.weight.data)
        nn.init.kaiming_normal_(self.lc2_fc.weight.data)
        nn.init.kaiming_normal_(self.lc3_fc.weight.data)
        nn.init.kaiming_normal_(self.gl_fc.weight.data)
        nn.init.kaiming_normal_(self.gl_fc2.weight.data)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, weight):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)

        lc_1 = self.lc1_fc(self.lc1_pool(x).view(x.size(0), -1))
        x = self.layer2(x)
        lc_2 = self.lc2_fc(self.lc2_pool(x).view(x.size(0), -1))
        x = self.layer3(x)
        lc_3 = self.lc3_fc(self.lc3_pool(x).view(x.size(0), -1))
        x = self.layer4(x)
        gl = self.gl_fc(self.gl_pool(x).view(x.size(0), -1))

        # feature fusion
        weight = weight.squeeze()
        if len(weight.size()) == 1:
            weight = weight.unsqueeze(0)
        weight1 = weight[:, 0:16].clone()
        weight2 = weight[:, 16:32].clone()
        weight3 = weight[:, 32:48].clone()
        weight4 = weight[:, 48:].clone()
        lc_11 = torch.mul(lc_1, weight1)
        lc_22 = torch.mul(lc_2, weight2)
        lc_33 = torch.mul(lc_3, weight3)
        gl_44 = self.gl_fc2(torch.mul(gl, weight4))
        vec = torch.cat((lc_11, lc_22, lc_33, gl_44), 1)

        out = {}
        out['fe_in_vec'] = vec

        return out


def resnet50_backbone(localScale, msfSize, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model_hyper.

    Args:
        pretrained (bool): If True, returns a model_hyper pre-trained on ImageNet
    """
    model = ResNetBackbone(localScale, msfSize, Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        save_model = model_zoo.load_url(model_urls['resnet50'])
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
    else:
        model.apply(weights_init_xavier)
    return model


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data)
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

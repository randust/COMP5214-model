import torchvision
import torch.nn as nn
import torch

def linear_stack(in_features, out_features, p = 0.3):
    return nn.Sequential(
        nn.Dropout(p),
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features),
        nn.ReLU(),
    )

"copied from resnet"
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
"end copy"

class PlantTraitsConv(nn.Module):
    def __init__(self):
        super().__init__()
        in_c = 3072
        out_c = 512
        
        backbone = torchvision.models.efficientnet_v2_m(weights='IMAGENET1K_V1')
        self.pretrained = backbone.features[:-2]
        dropout = nn.Dropout(p = 0.3, inplace=True)
        linear = nn.Linear(1280, 640, bias=True)
        relu = nn.ReLU()
        self.shared_tail = nn.Sequential(backbone.features[-1], backbone.avgpool, nn.Flatten(), dropout, linear, relu)

        for param in self.pretrained.parameters():
            param.requires_grad_(False)
        self.shared_conv = backbone.features[-2][:-3]

        self.shared_layer1_a = backbone.features[-2][-3].block[:-1]
        self.shared_layer1_b = backbone.features[-2][-3].block[-1]

        self.shared_layer2_a = backbone.features[-2][-2].block[:-1]
        self.shared_layer2_b = backbone.features[-2][-2].block[-1]

        self.shared_layer3_a = backbone.features[-2][-1].block[:-1]
        self.shared_layer3_b = backbone.features[-2][-1].block[-1]

        self.att1 = nn.ModuleList([self.att_layer(in_c, out_c // 4, out_c) for _ in range(6)])
        self.att2 = nn.ModuleList([self.att_layer(in_c + out_c, out_c // 4, out_c) for _ in range(6)])
        self.att3 = nn.ModuleList([self.att_layer(in_c + out_c, out_c // 4, out_c) for _ in range(6)])

        self.att1_downsp = self.conv_layer(out_c, out_c)
        self.att2_downsp = self.conv_layer(out_c, out_c)

        self.tail_layers = nn.ModuleList([self.tail_layer() for _ in range(6)])
            
    def forward(self, x):
        x = self.pretrained(x)
        x = self.shared_conv(x)

        h1_a = self.shared_layer1_a(x)
        h1_b = self.shared_layer1_b(h1_a)

        h2_a = self.shared_layer2_a(h1_b)
        h2_b = self.shared_layer2_b(h2_a)

        h3_a = self.shared_layer3_a(h2_b)
        h3_b = self.shared_layer3_b(h3_a)
        shared_features = self.shared_tail(h3_b)
        #attention block 1
        a1_mask = [att_i(h1_a) for att_i in self.att1]
        a1 = [a1_mask_i * h1_b for a1_mask_i in a1_mask]
        a1 = [self.att1_downsp(a1_i) for a1_i in a1]

        #attention block 2
        a2_mask = [att_i(torch.cat((h2_a, a1_i), dim=1)) for a1_i, att_i in zip(a1, self.att2)]
        a2 = [a2_mask_i * h2_b for a2_mask_i in a2_mask]
        a2 = [self.att2_downsp(a2_i) for a2_i in a2]

        #attention block 3
        a3_mask = [att_i(torch.cat((h3_a, a2_i), dim=1)) for a2_i, att_i in zip(a2, self.att3)]
        a3 = [a3_mask_i * h3_b for a3_mask_i in a3_mask]

        #attention transformation
        a4 = [tail_i(a3_i) for a3_i, tail_i in zip(a3, self.tail_layers)]
        return shared_features, a4
    
    def att_layer(self, in_channel, intermediate_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=intermediate_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(intermediate_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=intermediate_channel, out_channels=out_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channel),
            nn.Sigmoid())
    
    def conv_layer(self, in_channel, out_channel):
        downsample = nn.Sequential(conv1x1(in_channel, out_channel, stride=1),
                                   nn.BatchNorm2d(out_channel))
        return Bottleneck(in_channel, out_channel // 4, downsample=downsample)

    def tail_layer(self):
        backbone = torchvision.models.efficientnet_v2_m(weights = None)
        tail = backbone.features[-1]
        avgpool = backbone.avgpool
        dropout = nn.Dropout(p=0.3, inplace=True)
        linear = nn.Linear(1280, 640, bias=True)
        relu = nn.ReLU()
        return nn.Sequential(tail, avgpool, nn.Flatten(), dropout, linear, relu)
    
class PlantTraitsFeatures(nn.Module):
  def __init__(self, in_features = 163, out_features = 64, num_neurons = 326):
    super().__init__()
    self.linear_stack1 = linear_stack(in_features, num_neurons, p = 0)
    self.linear_stack2 = linear_stack(num_neurons, num_neurons)
    self.linear_stack3 = linear_stack(num_neurons, num_neurons)
    self.final_stack = linear_stack(num_neurons, out_features)

    self.att1 = nn.ModuleList([self.att_layer(num_neurons, num_neurons*2, num_neurons) for _ in range(6)])
    self.att2 = nn.ModuleList([self.att_layer(num_neurons + in_features, num_neurons*2, out_features) for _ in range(6)])

    self.att1_downsp = nn.ModuleList([nn.Sequential(linear_stack(num_neurons, num_neurons*2, p=0),
                                                    linear_stack(num_neurons*2, in_features)) for _ in range(6)])

  def forward(self, X):
    shared_out1 = self.linear_stack1(X) 
    
    shared_out2 = self.linear_stack2(shared_out1) + shared_out1

    shared_out3 = self.linear_stack3(shared_out2) + shared_out2

    shared_out = self.final_stack(shared_out3)

    a1_mask = [att_i(shared_out1) for att_i in self.att1]
    a1 = [a1_mask_i * shared_out2 for a1_mask_i in a1_mask]
    a1 = [downsp_i(a1_i) for a1_i, downsp_i in zip(a1, self.att1_downsp)]

    a2_mask = [att_i(torch.cat((shared_out3, a1_i), dim=1)) for a1_i, att_i in zip(a1, self.att2)]
    a2 = [a2_mask_i * shared_out for a2_mask_i in a2_mask]


    return shared_out, a2
  

  def att_layer(self, in_features, hidden_neurons, out_features):
    return nn.Sequential(
        nn.Linear(in_features, hidden_neurons),
        nn.BatchNorm1d(hidden_neurons),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_neurons, out_features),
        nn.BatchNorm1d(out_features),
        nn.Sigmoid(),
    )   
  
class PlantTraits(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = PlantTraitsConv()
        self.model2 = PlantTraitsFeatures()
        self.linear_stack1 = linear_stack(640+64, 326, p=0)
        self.linear_stack2 = linear_stack(326, 64)
        self.final_process = nn.ModuleList([nn.Sequential(linear_stack(64, 128, p=0), nn.Linear(128, 1)) for _ in range(6)])
        self.att1 = nn.ModuleList([self.att_layer(326+640+64, 326, 64) for _ in range(6)])

    def forward(self, x, y):
        shared1, attx = self.model1(x)
        shared2, atty = self.model2(y)

        shared_out1 = self.linear_stack1(torch.cat((shared1, shared2), dim=1))
        shared_out2 = self.linear_stack2(shared_out1)

        att_mask = [att_i(torch.cat((shared_out1, attx_i, atty_i),dim=1)) for att_i, attx_i, atty_i in zip(self.att1, attx, atty)]
        att = [shared_out2 * mask_i for mask_i in att_mask]

        return torch.cat([linear(att_i) for att_i, linear in zip(att, self.final_process)], dim=1)
    
    def att_layer(self, in_features, hidden_neurons, out_features):
        return nn.Sequential(
            nn.Linear(in_features, hidden_neurons),
            nn.BatchNorm1d(hidden_neurons),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_neurons, out_features),
            nn.BatchNorm1d(out_features),
            nn.Sigmoid(),
        )  
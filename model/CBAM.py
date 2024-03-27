import torch
import torch.nn as nn

"""
Defines Topt-CBAM, variants of CBAM
Use CBAM modules https://arxiv.org/abs/1807.06521
"""

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes) if bn else None
        self.relu = nn.LeakyReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    

class TopTPercentChannelGate(nn.Module):
    def __init__(self, gate_channels,percent_t, reduction_ratio=16, pool_types=['avg', 'max']):
        super(TopTPercentChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.percent_t = percent_t
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.LeakyReLU(inplace=True),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types
        
    def forward(self, x):
        b, c, _, _ = x.size()
        x_flatten = x.view(b, c, -1)
        
        # Calculate top T percent using self.percent_t
        top_t = int(round(x_flatten.size(2) * self.percent_t))

        channel_att_sum = None

        for pool_type in self.pool_types:
            if pool_type == 'avg':
                selected_values, _ = x_flatten.topk(top_t, dim=2)
                pool = selected_values.mean(dim=2, keepdim=True)
            elif pool_type == 'max':
                selected_values, _ = x_flatten.topk(top_t, dim=2)
                pool = selected_values.max(dim=2, keepdim=True)[0]
            else:
                raise ValueError("Invalid pool_type, choose between 'avg' and 'max'")
                
            channel_att_raw = self.mlp(pool)
            
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
                
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)

        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale

class Topt_CBAM(nn.Module):
    def __init__(self, gate_channels, percent_t, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(Topt_CBAM, self).__init__()

        self.ChannelGate = TopTPercentChannelGate(gate_channels,percent_t)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out



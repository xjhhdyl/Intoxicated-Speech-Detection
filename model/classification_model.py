import torch
import torch.nn as nn
from .att_modules import eca_layer
import torch.nn.functional as F


class ECABasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=(1, 1), downsample=None):
        super(ECABasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, (1, 1))
        self.bn2 = nn.BatchNorm2d(planes)
        self.eca = eca_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.eca(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


def conv3x3(in_planes, out_planes, stride=(1, 1)):
    """ 3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class rescrossSE(nn.Module):
    def __init__(self, channel, reduction=1):
        super(rescrossSE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        b, c, _, _, = x1.size()
        y1 = self.avg_pool(x1).view(b, c)
        y2 = self.avg_pool(x2).view(b, c)
        temp_y1 = y1
        y1 = self.fc(y2).view(b, c, 1, 1)
        y2 = self.fc(temp_y1).view(b, c, 1, 1)
        out1 = x1 * y1.expand_as(x1) + x1 * 0.5
        out2 = x1 * y2.expand_as(x2) + x2 * 0.5

        return out1, out2


class interattention(nn.Module):
    def __init__(self, all_channel):
        super(interattention, self).__init__()
        self.linear_e = nn.Conv2d(all_channel, all_channel, kernel_size=1, bias=False)
        self.channel = all_channel
        self.gate = nn.Conv2d(all_channel, 1, kernel_size=1, bias=False)
        self.gate_s = nn.Sigmoid()
        self.conv = nn.Conv2d(1, 1, kernel_size=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x1, x2):

        input_size = x1.size()[2:]
        # print(x1.size())
        all_dim = input_size[0] * input_size[1]
        x1_flat = x1.view(-1, x1.size()[1], all_dim)  # B, C, H*W
        x2_flat = x2.view(-1, x1.size()[1], all_dim)  # B,C,H*W
        x1_t = torch.transpose(x1_flat, 1, 2).contiguous()  # batch_size * dim * num
        # x1_corr = self.linear_e(x1_t) ### conv2d
        # print(x1_t.shape)
        A = torch.bmm(x1_t, x2_flat)
        # print(A.shape)
        A1 = F.softmax(A.clone(), dim=1)  # F.softmax(A.clone(), dim=1)
        B = F.softmax(torch.transpose(A, 1, 2), dim=1)

        x2_att = torch.bmm(x1_flat, A1).contiguous()  # S*Va = Zb #torch.bmm(x1_flat, A1).contiguous()
        x1_att = torch.bmm(x2_flat, B).contiguous()  # torch.bmm(x2_flat, B).contiguous()

        x1_att = x1_att.view(-1, x1.size()[1], input_size[0], input_size[1])
        x2_att = x2_att.view(-1, x1.size()[1], input_size[0], input_size[1])

        x1_mask = self.gate_s(x1_att)
        x2_mask = self.gate_s(x2_att)
        # x1_mask = self.gate(x1_mask)
        # x2_mask = self.gate(x2_mask)
        out1 = self.gate(x1_mask * x1)
        out2 = self.gate(x2_mask * x2)
        out = out1 + out2  # next to try element-wise product out = out1 * out2
        out = self.conv(out)
        # out = self.gate(x1_att + x2_att)
        return out

class mmWavoiceNet(nn.Module):
    def __init__(self, block, rescrossSEblock, interattentionblock, layers):
        super(mmWavoiceNet, self).__init__()
        self.inplanes = 8
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=[2, 1], padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, layers[0])  # block(self.inplanes, self.inplanes*2)
        self.layer2 = self._make_layer(block, 24, layers[1], stride=(2, 1))  # block(self.inplanes*2, self.inplanes*3)
        self.layer3 = self._make_layer(block, 24, layers[2], stride=(2, 1))  # block(self.inplanes*3, self.inplanes*4)
        self.rescrossSE1 = rescrossSEblock(self.inplanes)

        self.layer4 = self._make_layer(block, 24, layers[3])  # block(self.inplanes*4, self.inplanes*5)
        self.layer5 = self._make_layer(block, 24, layers[4])  # block(self.inplanes*5, self.inplanes*6)

        self.interattend1 = interattentionblock(self.inplanes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks=1, stride=(1, 1)):
        downsample = None
        if (stride[0] != 1 and len(stride) == 2) or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x2 = self.conv1(x2)

        x1 = self.bn1(x1)
        x2 = self.bn1(x2)

        x1 = self.relu(x1)
        x2 = self.relu(x2)

        x1 = self.layer1(x1)
        x2 = self.layer1(x2)

        x1 = self.layer2(x1)
        x2 = self.layer2(x2)

        x1 = self.layer3(x1)
        x2 = self.layer3(x2)

        x1, x2 = self.rescrossSE1(x1, x2)

        x1 = self.layer4(x1)
        x2 = self.layer4(x2)

        x1 = self.layer5(x1)
        x2 = self.layer5(x2)

        out = self.interattend1(x1, x2)
        return out


class mmWavoice(nn.Module):
    def __init__(self, rnnClassfication, mmwavoicenet):
        super(mmWavoice, self).__init__()
        self.rnnClassfication = rnnClassfication
        self.mmWavoiceNet = mmwavoicenet

    def forward(self, voice_batch_data, mmwave_batch_data, batch_label, is_training=True):
        voice_batch_data = voice_batch_data.unsqueeze(1)
        mmwave_batch_data = mmwave_batch_data.unsqueeze(1)
        mmWavoice_feature = self.mmWavoiceNet(voice_batch_data, mmwave_batch_data)
        mmWavoice_feature = mmWavoice_feature.squeeze(1)
        out = self.rnnClassfication(mmWavoice_feature)  # input : batch_data

        return out

class rnnClassfication(nn.Module):
    def __init__(self, input_feature_dim, hidden_dim, num_classes, rnn_uint="LSTM", dropout_rate=0.0):
        super(rnnClassfication, self).__init__()
        self.rnn_uint = getattr(nn, rnn_uint.upper())
        self.num_classes = num_classes
        self.BLSTM = self.rnn_uint(
            input_feature_dim * 2,
            hidden_dim,
            1,
            bidirectional=True,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.fc2 = nn.Linear(64, self.num_classes)
        self.dropout = nn.Dropout(0.4)

    def forward(self, input_x):
        batch_size = input_x.size(0)
        timestep = input_x.size(1)
        feature_dim = input_x.size(2)
        time_reduc = int(timestep / 2)
        input_xr = input_x.contiguous().view(batch_size, time_reduc, feature_dim * 2)
        x, (h_n, c_n) = self.BLSTM(input_xr)

        output_f = h_n[-2, :, :]  # batchsize * hidden_size 的向量
        output_b = h_n[-1, :, :]  # 如果是双向的则有两个h向量的输出；
        output = torch.cat([output_f, output_b], dim=-1)  # 将输出的两个tensor拼接在一起,dim=-1是纵向扩张
        out_fc1 = self.fc1(output)  # 输出放到两个全连接层,一个relu，一个softmax去算概率
        out_relu = F.relu(out_fc1)  # 每次通过线性层的时候都要激活，这里用的relu激活函数
        out = self.dropout(out_relu)
        out = self.fc2(out)
        return out
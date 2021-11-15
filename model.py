import copy
import torch
import torch.nn as nn

class first_conv_block(nn.Module):
    def __init__(self, in_c, hid_c, out_c, k_size, s_size, p_size):
        super(first_conv_block, self).__init__()
        self.conv1 = nn.Conv2d(in_c, hid_c, kernel_size=k_size, stride=s_size, padding=p_size)
        self.norm = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(hid_c, out_c, kernel_size=k_size, stride=s_size, padding=p_size)
        self.act = nn.ReLU()
    def forward(self, x):
        x = self.conv2(self.act(self.norm(self.conv1(x))))
        return x


class conv_block(nn.Module):
    def __init__(self, in_c, hid_c, out_c, k_size, s_size, p_size):
        super(conv_block, self).__init__()
        self.norm0 = nn.BatchNorm2d(in_c)
        self.conv1 = nn.Conv2d(in_c, hid_c, kernel_size=k_size, stride=s_size, padding=p_size)
        self.norm1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(hid_c, out_c, kernel_size=k_size, stride=1, padding=1)
        self.act = nn.ReLU()
    def forward(self, x):
        x = self.conv2(self.act(self.norm1(self.conv1(self.act(self.norm0(x))))))
        return x

class se_block(nn.Module):
    def __init__(self, in_c):
        super(se_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.line1 = nn.Linear(in_c, in_c//16, bias=False)
        self.line2 = nn.Linear(in_c//16, in_c, bias=False)

        self.act1 = nn.ReLU()
        self.act2 = nn.Sigmoid()
    def forward(self, x):
        b, c, _, _ = x.size()
        pre_x = x.clone()
        x = self.avg_pool(x).view(b, c)
        x = self.act2(self.line2(self.act1(self.line1(x)))).view(b, c, 1, 1)
        return pre_x * x.expand_as(pre_x)

class aspp_block(nn.Module):
    def __init__(self, in_c, out_c, k_size, s_size, p_size, di_size):
        super(aspp_block, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=k_size, stride=s_size, padding=p_size, dilation=di_size)
        self.act = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm2d(out_c)
    def forward(self, x):
        x = self.norm(self.act(self.conv(x)))
        return x

class ASPP(nn.Module):
    def __init__(self, in_c, out_c):
        super(ASPP, self).__init__()
        self.conv_block1 = aspp_block(in_c, out_c, 3, 1, 6, 6)
        self.conv_block2 = aspp_block(in_c, out_c, 3, 1, 12, 12)
        self.conv_block3 = aspp_block(in_c, out_c, 3, 1, 18, 18)

        self.output = nn.Conv2d(3 * out_c, out_c, 1)
        self._init_weights()

    def forward(self, x):
        x1 = self.conv_block1(x)
        x2 = self.conv_block2(x)
        x3 = self.conv_block3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        return self.output(out)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class att_block(nn.Module):
    def __init__(self, in_c, out_c, k_size, p_size):
        super(att_block, self).__init__()
        self.norm = nn.BatchNorm2d(in_c)
        self.act = nn.ReLU()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=k_size, padding=p_size)
    def forward(self, x):
        x = self.conv(self.act(self.norm(x)))
        return x


class attention(nn.Module):
    def __init__(self, in_encode, in_decode, out_c):
        super(attention, self).__init__()

        self.conv_encode = att_block(in_encode, out_c, 3, 1)
        self.encode_pool = nn.MaxPool2d(2, 2)

        self.conv_decode = att_block(in_decode, out_c, 3, 1)

        self.conv_att = att_block(out_c, 1, 1, 0)

    def forward(self, x1, x2):
        out = self.encode_pool(self.conv_encode(x1)) + self.conv_decode(x2)
        out = self.conv_att(out)
        return out * x2


class ResUNetPPlus(nn.Module):
    def __init__(self):
        super(ResUNetPPlus, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv_en1 = first_conv_block(3, 32, 32, 3, 1, 1)

        self.se2 = se_block(32)
        self.conv_en2_skip = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv_en2 = conv_block(32, 64, 64, 3, 2, 1)

        self.se3 = se_block(64)
        self.conv_en3_skip = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv_en3 = conv_block(64, 128, 128, 3, 2, 1)

        self.se4 = se_block(128)
        self.conv_en4_skip = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv_en4 = conv_block(128, 256, 256, 3, 2, 1)

        self.aspp1 = ASPP(256, 512)

        # Decoder
        self.att1 = attention(128, 512, 512)
        self.up1 = nn.Upsample(mode="bilinear", scale_factor=2)
        self.conv_de1_skip = nn.Conv2d(512+128, 256, kernel_size=3, stride=1, padding=1)
        self.conv_de1 = conv_block(512+128, 256, 256, 3, 1, 1)

        self.att2 = attention(64, 256, 256)
        self.up2 = nn.Upsample(mode="bilinear", scale_factor=2)
        self.conv_de2_skip = nn.Conv2d(256+64, 128, kernel_size=3, stride=1, padding=1)
        self.conv_de2 = conv_block(256+64, 128, 128, 3, 1, 1)

        self.att3 = attention(32, 128, 128)
        self.up3 = nn.Upsample(mode="bilinear", scale_factor=2)
        self.conv_de3_skip = nn.Conv2d(128+32, 64, kernel_size=3, stride=1, padding=1)
        self.conv_de3 = conv_block(128+32, 64, 64, 3, 1, 1)

        self.aspp2 = ASPP(64, 32)

        self.conv_out = nn.Conv2d(32, 2, 1)
        self.act_out = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv1(x) + self.conv_en1(x)  # (5, 32, 256, 256)
        x1 = self.se2(x1)  # (5, 32, 256, 256)

        x2 = self.conv_en2_skip(x1) + self.conv_en2(x1)  # (5, 64, 128, 128)
        x2 = self.se3(x2)  # (5, 64, 128, 128)

        x3 = self.conv_en3_skip(x2) + self.conv_en3(x2)  # (5, 128, 64, 64)
        x3 = self.se4(x3)  # (5, 128, 64, 64)

        x4 = self.conv_en4_skip(x3) + self.conv_en4(x3)  # (5, 256, 32, 32)

        aspp1 = self.aspp1(x4)  # (5, 512, 32, 32)

        att1 = self.att1(x3, aspp1)  # (5, 512, 32, 32)
        x5 = torch.cat([self.up1(att1), x3], dim=1)  # (5, 640, 64, 64)
        x5 = self.conv_de1_skip(x5) + self.conv_de1(x5)  # (5, 256, 64, 64)

        att2 = self.att2(x2, x5)  # (5, 256, 64, 64)
        x6 = torch.cat([self.up2(att2), x2], dim=1)  # (5, 320, 128, 128)
        x6 = self.conv_de2_skip(x6) + self.conv_de2(x6)  # (5, 128, 128, 128)

        att3 = self.att3(x1, x6)  # (5, 128, 128, 128)
        x7 = torch.cat([self.up3(att3), x1], dim=1)  # (5, 160, 256, 256)
        x7 = self.conv_de3_skip(x7) + self.conv_de3(x7)  # (5, 64, 256, 256)

        x8 = self.aspp2(x7)  # (5, 32, 256, 256)
        out = self.act_out(self.conv_out(x8))  # (5, 2, 256, 256)

        return out



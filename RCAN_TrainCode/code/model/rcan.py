from model import common

import torch.nn as nn
import torch
import numpy as np

import cv2

def make_model(args, parent=False):
    return RCAN(args)

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

## Residual Channel Attention Network (RCAN)
class RCAN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(RCAN, self).__init__()
        
        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction 
        scale = args.scale[0]
        act = nn.ReLU(True)
        
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        # self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)

        # add deal graycode layer
        # self.models_graycode = commen.convert_graycode

        # define head module
        # modules_head = [conv(args.n_colors, n_feats, kernel_size)]  # (3,64,3 )
        modules_head = [conv(8, n_feats, kernel_size)]  # (3,64,3 )

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            # conv(n_feats, args.n_colors, kernel_size)]
            conv(n_feats, 8, kernel_size)]
        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):

        # print(type(x.cpu().detach().data.numpy()))
        # torch.size([16,3,48,48])

        x0 = (x.cpu().detach().data.numpy()).astype("uint8")    # x转numpy

        x1 = np.zeros([x0.shape[0], x0.shape[2], x0.shape[3]], dtype='uint8')

        for i in range(x0.shape[0]):
            img = x0[i, :, :, :]
            img = np.transpose(img,(1,2,0))  
            r, g, b = cv2.split(img)
            img = cv2.merge([b, g, r])
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
            img_y = img[:, :, 0]
            x1[i, :, :] = img_y

        img = x1    # [16, 48, 48]的y通道图像

        # 转十进制格雷码
        img_new = (img.astype("uint8") / 2).astype("uint8")    
        img = img.astype("uint8")
        img_new = img ^ img_new # [16,48,48]

        [h, w] = img_new.shape[1],img_new.shape[2],

        image = np.empty((x0.shape[0], 8, h, w), dtype=np.uint8)  # 存 余数 image[16,8,48,48]

        for i in range(8):
            image[:, i, :, :] = img_new % 2  # 转格雷码8维图像
            # print(image[:, :, i])
            img_new = img_new // 2  

        x = torch.from_numpy(np.ascontiguousarray(image)).float()
        # x = torch.FloatTensor(x)
        x = x.cuda()

        # print(x.shape)

        # print(self.sub_mean)

        # x = self.sub_mean(x)

        x = self.head(x)
        res = self.body(x)
        res += x

        x = self.tail(res)
        # x = self.add_mean(x)
        # print(x.shape)

        return x

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))

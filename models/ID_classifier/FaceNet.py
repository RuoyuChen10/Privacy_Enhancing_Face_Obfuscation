# -*- coding: utf-8 -*-  

"""
Created on 2022/04/22

Author: Ruoyu Chen
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from .iresnet import iresnet50

class FaceNet(nn.Module):
    """
    FaceNet
    @Author: Ruoyu Chen
    """
    def __init__(self, 
        net = "VGGFace2",
        include_top = True
    ):
        super().__init__()
        self.img_size = 112
        self.net = net
        self.include_top = include_top
        self.init_model(self.net)

        self.softmax = nn.Softmax(dim=1)

    def init_model(self, net):
        """
        Init
            net: [VGGFace2, CelebA]
        """
        if net == "VGGFace2":
            self.model = iresnet50(people_num = 8631)
            self.load_pretrained(weight="./models/ID_classifier/ckpt/vggface2/ArcFace-r50-8631.pkl")
        elif net == "CelebA":
            self.model = iresnet50(people_num = 10177)
            self.load_pretrained(weight="./models/ID_classifier/ckpt/celeba/ArcFace-r50-10177.pkl")
        else:
            raise Exception('net only support \"VGGFace2\" and \"CelebA\", key {} is wrong'.format(net))

        self.model.eval()   # Freeze
    
    def load_pretrained(self, weight):
        """
        Load pretrained model
        """
        # No related
        model_dict = self.model.state_dict()
        pretrained_param = torch.load(weight, map_location=torch.device('cpu'))
        try:
            pretrained_param = pretrained_param.state_dict()
        except:
            pass

        new_state_dict = OrderedDict()
        for k, v in pretrained_param.items():
            if k in model_dict:
                new_state_dict[k] = v
                print("Load parameter {}".format(k))
            elif k[7:] in model_dict:
                new_state_dict[k[7:]] = v
                print("Load parameter {}".format(k[7:]))

        model_dict.update(new_state_dict)
        self.model.load_state_dict(model_dict)
        print("Success load pre-trained face model {}".format(weight))
    
    def forward(self, x):
        # input size
        if x.size(-1) != self.img_size:
                x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear')
        
        output = self.model(x, self.include_top)

        if self.include_top:
            output = self.softmax(output)

        return output

    

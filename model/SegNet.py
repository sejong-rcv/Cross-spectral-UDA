import torch
import torch.nn as nn

from .generator import Generator
from .Decoder import Decoder
from utils.initial_utils import init_net
from utils.utils import find_norm


class SegNet(nn.Module):
    ## resnet pretrained version

    def __init__(self, args, num_layers=50):
        super(SegNet, self).__init__()

        self.num_layers = num_layers
        self.inplanes = 2048

        norm = find_norm(args.norm)
        
        self.net_G_rgb = init_net(Generator(sensor='rgb', num_layers=self.num_layers), init_type=args.init_type,
                                  net_type='encoder_rgb')
        self.net_G_thermal = init_net(Generator(sensor='thermal', num_layers=self.num_layers), init_type=args.init_type,
                                      net_type='encoder_th')
        self.decoder = init_net(Decoder(args.num_classes, self.inplanes, norm_layer=norm), init_type=args.init_type,
                                net_type='decoder') 
            
        #Following RTFNet, we use pretrained encoder and initialize decoder with xavier method.



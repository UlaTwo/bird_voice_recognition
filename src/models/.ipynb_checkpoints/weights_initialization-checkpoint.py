import torch.nn as nn

""" Funkcja implementująca inicjalizację wag warstw sieci 
    według metody Kaiming'a 
    (która dobrze współdziała z funkcją ReLu jako funkcją aktywacji)
"""
def weights_init_kaiming(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

"""Funkcja uruchamiająca inicjalizacja wag warstw"""
def init_weights(net):
    net.apply(weights_init_kaiming)

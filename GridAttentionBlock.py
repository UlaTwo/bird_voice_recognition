import torch
from torch import nn
from torch.nn import functional as F
from weights_initialization import init_weights

class GridAttentionBlock(torch.nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels,
                 sub_sample_factor, use_phi, use_theta, use_psi):
        
        super(GridAttentionBlock, self).__init__()

        self.dimension = 2
        
        self.sub_sample_factor = sub_sample_factor if isinstance(sub_sample_factor, tuple) else tuple([sub_sample_factor])*dimension
        
        self.sub_sample_kernel_size = sub_sample_factor

        # Number of channels
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels

                
        conv = nn.Conv2d
        bn = nn.BatchNorm2d


        # initialise functions
        # Theta^T * x_ij + Phi^T * gating_signal + bias
        self.theta = lambda x: x
        self.psi = lambda x: x
        self.phi = lambda x: x
        self.nl1 = lambda x: x

        # pytanie - czy zostawić ten warunek, czy zmienić na stałe?
        if use_theta:
            self.theta = conv(in_channels=self.in_channels, out_channels=self.inter_channels,
                                 kernel_size=self.sub_sample_kernel_size, stride=self.sub_sample_factor, padding=0, bias=False)

        if use_phi:
            self.phi = conv(in_channels=self.gating_channels, out_channels=self.inter_channels,
                               kernel_size=self.sub_sample_kernel_size, stride=self.sub_sample_factor, padding=0, bias=False) #tutaj pytani - czy true, czy false

        if use_psi:
            self.psi = conv(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)


        self.nl1 = lambda x: F.relu(x, inplace=True)

        self.attention_gate_function = self._concatenation

        # Initialise weights
        for m in self.children():
            init_weights(m, init_type='kaiming')


    def forward(self, x, g):
        
        output = self.attention_gate_function(x, g)
        return output

    def _concatenation(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        #############################
        # compute first equation of Attention Gate - sum x and g, than nonlinear sigmoid function
        # nl(theta.x + phi.g + bias)

        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # upsample g to the size of theta
        phi_g = F.interpolate(self.phi(g), size=theta_x_size[2:], mode='bilinear', align_corners=True)

        f = theta_x + phi_g
        f = self.nl1(f)  #sigmoid1 - relu

        psi_f = self.psi(f)  #conv2d
        
        #end of the first equation

        ############################################

        #Normalization
        psi_f_flat = psi_f.view(batch_size, 1, -1)
        ss = psi_f_flat.shape
        psi_f_max = torch.max(psi_f_flat, dim=2)[0].view(ss[0], ss[1], 1)
        psi_f_min = torch.min(psi_f_flat, dim=2)[0].view(ss[0], ss[1], 1)

        sigm_psi_f = (psi_f_flat - psi_f_min) / (psi_f_max - psi_f_min).expand_as(psi_f_flat)
        sigm_psi_f = sigm_psi_f.view(batch_size, 1, *theta_x_size[2:])
        

        # sigm_psi_f is attention map - upsample the attentions and multiply
        sigm_psi_f = F.interpolate(sigm_psi_f, size=input_size[2:], mode='bilinear', align_corners=True)
        y = sigm_psi_f.expand_as(x) * x

        return y, sigm_psi_f

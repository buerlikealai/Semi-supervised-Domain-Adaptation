import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

def partial_padding(fs,h_,w_,*args,**kwargs):
    slide_window = fs*fs #p1
    mask = torch.ones([1,1,h_,w_], dtype=torch.float32,device='cuda')# torch.tensor are created on cpu
    Filter = torch.ones([1,1,fs,fs],dtype=torch.float32,device='cuda')
    
    padding = nn.ZeroPad2d(1)
    mask = padding(mask)
    update_mask = F.conv2d(mask,Filter)#p0
        
    mask_ratio = slide_window/(update_mask+1e-8)
    update_mask = torch.clamp(update_mask, min=0.0, max=1.0)
    mask_ratio = mask_ratio*update_mask
    
    return mask_ratio


class ResDown(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=7,
                 scale=4 ): 
        super().__init__()
        self.scale = scale

        self.down = nn.Sequential(
            nn.ZeroPad2d(2),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=scale))
        #self.down[0].weight = nn.Parameter(
            #nn.init.kaiming_normal_(torch.empty((out_channels,in_channels,scale,scale),dtype=torch.float32)),
            #requires_grad = True) # cause problem
        
        self.mask_ratio = partial_padding(3,64,64)
        
    def forward(self, x):
        #print('input feature size',x.shape)
        x = self.down(x)
        #print('before partial padding:output of first convertional layer',x.shape)
        x = x*self.mask_ratio
        x = F.leaky_relu(x,inplace=True)
        #print('the output of down sampling',x.shape)
        return x
    
class resblock(nn.Module):
    def __init__(self,
               ch_io,
               fs=3):
        super().__init__()
        self.fs = fs
        self.ch_io = ch_io
        
        Conv = partial(nn.Conv2d, ch_io, ch_io // 4, 3)#, padding_mode='replicate')
        
        self.conv1 = nn.Sequential(#nn.ZeroPad2d(1),
            Conv(dilation=1,padding=1))
        self.conv2 = nn.Sequential(#nn.ZeroPad2d(1),
            Conv(dilation=2,padding=2))
        self.conv3 = nn.Sequential(#nn.ZeroPad2d(1),
            Conv(dilation=3,padding=3))
        self.conv4 = nn.Sequential(#nn.ZeroPad2d(1),
            Conv(dilation=4,padding=4))
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                #nn.init.constant_(m.bias, 0)
                #m.bias.requires_grad = True
        
        self.merge_conv = nn.Sequential(nn.Conv2d(ch_io, ch_io, kernel_size=1),
                                        nn.Dropout2d(p=0.5, inplace=True))
        
        self.mask_ratio = partial_padding(3,64,64)
        
    def forward(self,x):
        #print('the input for resblock',x.shape)
        a1 = self.conv1(x)
        #print('the 1st convolutional layer in residual block output',a1.shape)
        
        a1 = a1*self.mask_ratio
        #print('afetr partial convolution',a1.shape)
        a1 = F.leaky_relu(a1,inplace=True)
        #print('output of 1st res block',a1.shape)
        
        a2 = self.conv2(x)
        #print('the 2nd convolutional layer in residual block output',a2.shape)
        a2 = a2*self.mask_ratio
        a2 = F.leaky_relu(a2,inplace=True)
        #print('output of 2nd res block',a2.shape)
        
        a3 = self.conv3(x)
        #print('the 3rd convolutional layer in residual block output',a3.shape)
        a3 = a3*self.mask_ratio
        a3 = F.leaky_relu(a3,inplace=True)
        #print('output of 3rd res block',a3.shape)
        
        a4 = self.conv4(x)
        #print('the 4th convolutional layer in residual block output',a4.shape)
        a4 = a4*self.mask_ratio
        a4 = F.leaky_relu(a4,inplace=True)
        #print('output of 4th res block',a4.shape)
        
        A = torch.cat((a1, a2, a3, a4), dim=1)
        #print('the concated A',A.shape)
        merged = self.merge_conv(A)
        #print('the merge_conv(A)',merged.shape)
        merged = merged*self.mask_ratio
        merged = F.leaky_relu(merged,inplace=True)
        #print('the merge_conv(A) after activation function',merged.shape)
        return x + merged

class ResUp(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 scale=4, 
                 bilinear=False):
        super().__init__()
        self.scale = scale
   
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=scale,
                                   stride=scale,output_padding=(2,2)),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.0, inplace=True))
    
    
        self.up[0].weight = nn.Parameter(
            nn.init.kaiming_normal_(torch.empty((in_channels,out_channels,scale,scale),dtype=torch.float32)),
            requires_grad = True)
    
        #self.up[0].bias = nn.Parameter(nn.init.zeros_(torch.empty((out_channels),dtype=torch.float32)),
                                       #requires_grad = True)       
        
    def forward(self, x):
     
        return self.up(x)
    
class RNet(nn.Module):
    def __init__(self, n_channels, n_classes, fn=88, kernel_size=7, scale=4):
        super(RNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.down = ResDown(n_channels, fn, kernel_size,scale)
        self.mid = nn.Sequential(*[resblock(fn) for i in range(10)]) # resblock(in_channel, kernel_size)
        self.up = ResUp(fn, fn // 2, scale)
        
        self.outc = nn.Conv2d(fn // 2, n_classes, kernel_size=3)  # logistic regression

    def forward(self, x):#, ret_probas=False, ret_preds=False):
        #torch.cuda.synchronize()
        #net_forward_start = timer()
        
        #print('before downsampling/oroginal input',x.shape)
        xD = self.down(x)
        #print('before residual block',xD.shape)
        xM = self.mid(xD)
        #print('after residual block, before up sampling',xM.shape)
        xU = self.up(xM)  # tanh?
        #print('after up sampling',xU.shape)

        logits = self.outc(xU) # sort of like the probability
        #print('output layer',logits.shape)
        # output for each sample a vector of length N classes, 
        # where the value for index is the logit of that sample being of class i
        
        probas = F.softmax(logits, 1) #if ret_probas else None # the normalized probabilities can be summed up to 1
        preds = torch.argmax(logits, 1) #if ret_preds else None
        # returns the index that the model predicts as having the highest probability of being class label
        # this is taken as predicted class label
        
        return logits, probas, preds#, net_forward_end

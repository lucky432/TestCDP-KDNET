from turtle import shape
import torch
import torch.nn as nn
from correlation_package.correlation import Correlation
import numpy as np
from utils import flow_warp

from GOCor import local_gocor
from GOCor.optimizer_selection_functions import define_optimizer_local_corr
from utils.losses import GradNorm, NCC, LNCC
from utils import warp_image


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):   
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, 
                        padding=padding, dilation=dilation, bias=True),
            nn.LeakyReLU(0.1))

def predict_flow(in_planes):
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=True)

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)



class PWCNet(nn.Module):
    """
    PWC-DC net. add dilation convolution and densenet connections

    """
    def __init__(self, md=4):
        """
        input: md --- maximum displacement (for correlation. default: 4), after warpping

        """
        super(PWCNet,self).__init__()
        self.conv1a  = conv(1,   16, kernel_size=3, stride=1)
        self.conv1aa = conv(16,  16, kernel_size=3, stride=1)
        self.conv1b  = conv(16,  16, kernel_size=3, stride=1)
        self.conv3a  = conv(16,  64, kernel_size=3, stride=2)
        self.conv3aa = conv(64,  64, kernel_size=3, stride=1)
        self.conv3b  = conv(64,  64, kernel_size=3, stride=1)
        self.conv6aa = conv(64, 196, kernel_size=3, stride=2)
        self.conv6a  = conv(196,196, kernel_size=3, stride=1)
        self.conv6b  = conv(196,196, kernel_size=3, stride=1)

        # self.corr    = Correlation(pad_size=md, kernel_size=1, max_displacement=md, stride1=1, stride2=1, corr_multiply=1)
        self.leakyRELU = nn.LeakyReLU(0.1)
        
        nd = (2*md+1)**2
        dd = np.cumsum([128,128,96,64,32])

        od = nd
        self.conv6_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv6_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv6_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv6_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv6_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)        
        self.predict_flow6 = predict_flow(od+dd[4])
        self.deconv6 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        self.upfeat6 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1) 
        
        od = nd+64+4
        self.conv3_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv3_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv3_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv3_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv3_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow3 = predict_flow(od+dd[4]) 
        self.deconv3 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        self.upfeat3 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1) 
        
        od = nd+16+4
        self.conv2_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv2_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv2_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv2_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv2_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow2 = predict_flow(od+dd[4]) 
        self.deconv2 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        self.predict_flow2 = predict_flow(od+dd[4]) 
        self.deconv2 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        
        self.dc_conv1 = conv(od+dd[4], 128, kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc_conv2 = conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2)
        self.dc_conv3 = conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4)
        self.dc_conv4 = conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8)
        self.dc_conv5 = conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc_conv6 = conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc_conv7 = predict_flow(32)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

        local_gocor_arguments = {'optim_iter': 3}

        initializer_6 = local_gocor.LocalCorrSimpleInitializer()
        optimizer_6 = define_optimizer_local_corr(local_gocor_arguments)
        self.local_corr_6 = local_gocor.LocalGOCor(filter_initializer=initializer_6, filter_optimizer=optimizer_6)

        initializer_5 = local_gocor.LocalCorrSimpleInitializer()
        optimizer_5 = define_optimizer_local_corr(local_gocor_arguments)
        self.local_corr_5 = local_gocor.LocalGOCor(filter_initializer=initializer_5, filter_optimizer=optimizer_5)

       
        initializer_2 = local_gocor.LocalCorrSimpleInitializer()
        optimizer_2 = define_optimizer_local_corr(local_gocor_arguments)
        self.local_corr_2 = local_gocor.LocalGOCor(filter_initializer=initializer_2, filter_optimizer=optimizer_2)

    def forward(self,input,train=True):
        self.training = train
        im1 = input[:,0:1,:,:].clone()
        im2 = input[:,1:2,:,:].clone()
        
        c11 = self.conv1b(self.conv1aa(self.conv1a(im1)))
        c21 = self.conv1b(self.conv1aa(self.conv1a(im2)))
        
        c12 = self.conv3b(self.conv3aa(self.conv3a(c11)))
        c22 = self.conv3b(self.conv3aa(self.conv3a(c21)))

        c13 = self.conv6b(self.conv6a(self.conv6aa(c12)))
        c23 = self.conv6b(self.conv6a(self.conv6aa(c22)))


        corr6 = self.local_corr_6(c13, c23) 
        corr6 = self.leakyRELU(corr6)   


        x = torch.cat((self.conv6_0(corr6), corr6),1)
        x = torch.cat((self.conv6_1(x), x),1)
        x = torch.cat((self.conv6_2(x), x),1)
        x = torch.cat((self.conv6_3(x), x),1)
        x = torch.cat((self.conv6_4(x), x),1)
        flow6 = self.predict_flow6(x)
        up_flow6 = self.deconv6(flow6)
        up_feat6 = self.upfeat6(x)

        
        warp3 = flow_warp(c22, up_flow6*1.25)
        corr3 = self.local_corr_5(c12, warp3)  
        corr3 = self.leakyRELU(corr3)
        x = torch.cat((corr3, c12, up_flow6, up_feat6), 1)
        x = torch.cat((self.conv3_0(x), x),1)
        x = torch.cat((self.conv3_1(x), x),1)
        x = torch.cat((self.conv3_2(x), x),1)
        x = torch.cat((self.conv3_3(x), x),1)
        x = torch.cat((self.conv3_4(x), x),1)
        flow3 = self.predict_flow3(x)
        up_flow3 = self.deconv3(flow3)
        up_feat3 = self.upfeat3(x)



        #-----------------------------------------------------#
        #--------------------testnet_mini---------------------#
        #-----------------------------------------------------#

        # rf1 = torch.zeros_like(im1)
        # rf2 = torch.zeros_like(im2)
        # rf1[:,0:1,:,:] = input[:,0:1,:,:].clone()
        # rf1[:,1:2,:,:] = input[:,0:1,:,:].clone()
        # rf2[:,0:1,:,:] = input[:,2:3,:,:].clone()
        # rf2[:,1:2,:,:] = input[:,2:3,:,:].clone()
        # rf1 = rf1.type_as(im1)
        # rf2 = rf2.type_as(im2)

        #########################################
        rf1 = input[:,0:1,:,:].clone()
        rf2 = input[:,1:2,:,:].clone()

        c11_ = self.conv1b(self.conv1aa(self.conv1a(rf1)))
        c21_ = self.conv1b(self.conv1aa(self.conv1a(rf2)))
       
        
        warp2 = flow_warp(c11_, up_flow3*2.5) 
        corr2 = self.local_corr_2(c21_, warp2)
        corr2 = self.leakyRELU(corr2)

        x = torch.cat((corr2, c11_, up_flow3, up_feat3), 1)
        x = torch.cat((self.conv2_0(x), x),1)
        x = torch.cat((self.conv2_1(x), x),1)
        x = torch.cat((self.conv2_2(x), x),1)
        x = torch.cat((self.conv2_3(x), x),1)
        x = torch.cat((self.conv2_4(x), x),1)
        flow2 = self.predict_flow2(x)
        #########################################

        # warp2 = flow_warp(c22, up_flow3*5.0) 
        # corr2 = self.corr(c12, warp2)
        # corr2 = self.leakyRELU(corr2)
        # x = torch.cat((corr2, c12, up_flow3, up_feat3), 1)
        # x = torch.cat((self.conv2_0(x), x),1)
        # x = torch.cat((self.conv2_1(x), x),1)
        # x = torch.cat((self.conv2_2(x), x),1)
        # x = torch.cat((self.conv2_3(x), x),1)
        # x = torch.cat((self.conv2_4(x), x),1)
        # flow2 = self.predict_flow2(x)

        x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
        flow2 += self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))

        if self.training:
            return flow2,flow3,flow6
        else:
            return flow2
        
    # def lossCal(self,pre,post,output):
    #     warped_img = warp_image(post, output)
    #     self.loss_similarity = LNCC(warped_img,pre,[51,11])
    #     # self.loss_similarity = NCC(self.warped_img,self.pre)
    #     self.loss_smooth = GradNorm(output)
    #     self.loss_total = self.w_similarity*(1-self.loss_similarity) + self.w_smooth*self.loss_smooth

    #     return self.loss_total
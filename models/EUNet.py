import torch
import torch.nn as nn
import torchvision.models as models

from models.modules import LCA,ASM,GCM_up,GCM,CrossNonLocalBlock


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1):
        super(DecoderBlock, self).__init__()

        self.conv1 = ConvBlock(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.conv2 = ConvBlock(in_channels // 4, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.upsample(x)
        return x


class SideoutBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SideoutBlock, self).__init__()

        self.conv1 = ConvBlock(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.dropout = nn.Dropout2d(0.1)

        self.conv2 = nn.Conv2d(in_channels // 4, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)

        return x


class EUNet(nn.Module):
    def __init__(self, num_classes):
        super(EUNet, self).__init__()

        resnet = models.resnet34(pretrained=True)
       
        # Encoder
        self.encoder1_conv = resnet.conv1
        self.encoder1_bn = resnet.bn1
        self.encoder1_relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.encoder2 = resnet.layer1
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4


        # Decoder
        self.decoder5 = DecoderBlock(in_channels=512, out_channels=512)
        self.decoder4 = DecoderBlock(in_channels=1024, out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=512, out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=256, out_channels=64)
        self.decoder1 = DecoderBlock(in_channels=192, out_channels=64)

        self.outconv = nn.Sequential(ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
                                      nn.Dropout2d(0.1),
                                      nn.Conv2d(32, num_classes, 1))

        self.outenc = ConvBlock(512,256,kernel_size=1, stride=1,padding=0)

        # Sideout
        self.sideout2 = SideoutBlock(64, 1)
        self.sideout3 = SideoutBlock(128, 1)
        self.sideout4 = SideoutBlock(256, 1)
        self.sideout5 = SideoutBlock(512, 1)

   

        # global context module
        self.gcm_up = GCM_up(256,64)
        self.gcm_e5 = GCM_up(256, 256)#3
        
        self.gcm_e4 = GCM_up(256, 128)#2
        self.gcm_e3 = GCM_up(256, 64)#1
        self.gcm_e2 = GCM_up(256, 64)#0


        # adaptive selection module
        self.asm4 = ASM(512, 1024)
        self.asm3 = ASM(256, 512)
        self.asm2 = ASM(128, 256)
        self.asm1 = ASM(64, 192)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) 
        self.up2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True) 
        self.up3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True) 
        self.up4 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

        self.lca_cross_1 = CrossNonLocalBlock(512,256,256)
        self.lca_cross_2 = CrossNonLocalBlock(1024,128,128)
        self.lca_cross_3 = CrossNonLocalBlock(512,64,64)
        self.lca_cross_4 = CrossNonLocalBlock(256,64,64)

    def forward(self, x):
        e1 = self.encoder1_conv(x) 
        e1 = self.encoder1_bn(e1)
        e1 = self.encoder1_relu(e1)
        e1_pool = self.maxpool(e1)  
        e2 = self.encoder2(e1_pool)
        e3 = self.encoder3(e2) 
        e4 = self.encoder4(e3)  
        e5 = self.encoder5(e4)  
        e_ex = self.outenc(e5)
        
        global_contexts_up = self.gcm_up(e_ex)

        
        d5 = self.decoder5(e5)  
        out5 = self.sideout5(d5)
        lc4 = self.lca_cross_1(d5,e4)
        gc4 = self.gcm_e5(e_ex)
        gc4 = self.up1(gc4)
        
        
        comb4 = self.asm4(lc4, d5, gc4)

        d4 = self.decoder4(comb4) 
        out4 = self.sideout4(d4)
        lc3 = self.lca_cross_2(comb4,e3)
        gc3 = self.gcm_e4(e_ex)
        gc3 = self.up2(gc3)
        
 
        comb3 = self.asm3(lc3, d4, gc3)
        

        d3 = self.decoder3(comb3)
        out3 = self.sideout3(d3)
        lc2= self.lca_cross_3(comb3,e2)
        gc2 = self.gcm_e3(e_ex)
        gc2 = self.up3(gc2)
        
        comb2 = self.asm2(lc2, d3, gc2)

        d2 = self.decoder2(comb2)  
        out2 = self.sideout2(d2)
        lc1 = self.lca_cross_4(comb2,e1)
        gc1 = self.gcm_e2(e_ex)
        gc1 = self.up4(gc1)
       
        comb1 = self.asm1(lc1, d2, gc1)

        d1 = self.decoder1(comb1) 
        out1 = self.outconv(d1)  

        return torch.sigmoid(out1), torch.sigmoid(out2), torch.sigmoid(out3), \
            torch.sigmoid(out4), torch.sigmoid(out5)

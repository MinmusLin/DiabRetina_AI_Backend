import os
import torch
from torch import nn
from .Layers import SN_CS_Parallel_Attention_block, SN_PostRes2d
from torch import optim
from torch.backends import cudnn
from torch.nn.functional import interpolate
from .Layers import SwitchNorm2d

class CAUNet(nn.Module):

    def __init__(self, input_channel=3, output_channel=1, bn_momentum=0.2, feature=False):
        super(CAUNet, self).__init__()
        self.original_size = 640
        self.feature = feature
        self.nb_filter = [32, 64, 128, 256, 512]
        self.preBlock = nn.Sequential(
            nn.Conv2d(input_channel, self.nb_filter[0], kernel_size=3, padding=1),
            SwitchNorm2d(self.nb_filter[0], momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.nb_filter[0], self.nb_filter[0], kernel_size=3, padding=1),
            SwitchNorm2d(self.nb_filter[0], momentum=bn_momentum),
            nn.ReLU(inplace=True),
        )
        self.outBlock = nn.Sequential(
            nn.Conv2d(48, 24, kernel_size=3, padding=1),
            SwitchNorm2d(24, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, kernel_size=3, padding=1),
            SwitchNorm2d(24, momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )
        self.pred = nn.Sequential(
            nn.Conv2d(self.nb_filter[0], 12, kernel_size=1),
            SwitchNorm2d(12, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(12, output_channel, kernel_size=1),
            nn.Sigmoid()
        )
        num_blocks_forw = [2, 2, 3, 4]
        self.featureNum_forw = self.nb_filter
        for i in range(len(num_blocks_forw)):
            blocks = []
            for j in range(num_blocks_forw[i]):
                if j == 0:
                    blocks.append(SN_PostRes2d(self.featureNum_forw[i], self.featureNum_forw[i + 1]))
                else:
                    blocks.append(SN_PostRes2d(self.featureNum_forw[i + 1], self.featureNum_forw[i + 1]))
            setattr(self, 'forw' + str(i + 1), nn.Sequential(*blocks))
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.back1 = nn.Sequential(
            SN_PostRes2d(992, self.nb_filter[3]),
            SN_PostRes2d(self.nb_filter[3], self.nb_filter[3]),
        )
        self.back2 = nn.Sequential(
            SN_PostRes2d(608, self.nb_filter[2]),
            SN_PostRes2d(self.nb_filter[2], self.nb_filter[2]),
        )
        self.back3 = nn.Sequential(
            SN_PostRes2d(416, self.nb_filter[1]),
            SN_PostRes2d(self.nb_filter[1], self.nb_filter[1]),
        )
        self.back4 = nn.Sequential(
            SN_PostRes2d(320, self.nb_filter[0]),
            SN_PostRes2d(self.nb_filter[0], self.nb_filter[0]),
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(self.nb_filter[4], self.nb_filter[3], kernel_size=2, stride=2),
            SwitchNorm2d(self.nb_filter[3], momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(self.nb_filter[3], self.nb_filter[2], kernel_size=2, stride=2),
            SwitchNorm2d(self.nb_filter[2], momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(self.nb_filter[2], self.nb_filter[1], kernel_size=2, stride=2),
            SwitchNorm2d(self.nb_filter[1], momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(self.nb_filter[1], self.nb_filter[0], kernel_size=2, stride=2),
            SwitchNorm2d(self.nb_filter[0], momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )
        self.up4 = nn.Sequential(
            nn.Conv2d(self.nb_filter[4], self.nb_filter[3], kernel_size=3),
            SwitchNorm2d(self.nb_filter[3], momentum=bn_momentum),
            nn.ReLU(inplace=True),
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(self.nb_filter[3], self.nb_filter[2], kernel_size=3),
            SwitchNorm2d(self.nb_filter[2], momentum=bn_momentum),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(self.nb_filter[2], self.nb_filter[1], kernel_size=3),
            SwitchNorm2d(self.nb_filter[1], momentum=bn_momentum),
            nn.ReLU(inplace=True),
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(self.nb_filter[1], self.nb_filter[0], kernel_size=3),
            SwitchNorm2d(self.nb_filter[0], momentum=bn_momentum),
            nn.ReLU(inplace=True),
        )
        self.out_conv1 = nn.Sequential(
            nn.ConvTranspose2d(self.nb_filter[3], output_channel, kernel_size=8, stride=8),
            nn.Sigmoid()
        )
        self.out_conv2 = nn.Sequential(
            nn.ConvTranspose2d(self.nb_filter[2], output_channel, kernel_size=4, stride=4),
            nn.Sigmoid()
        )
        self.out_conv3 = nn.Sequential(
            nn.ConvTranspose2d(self.nb_filter[1], output_channel, kernel_size=2, stride=2),
            nn.Sigmoid()
        )
        self.pool_preblock_gate1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool_preblock_gate2 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.pool_preblock_gate3 = nn.MaxPool2d(kernel_size=8, stride=8)
        self.pool_out1_gate2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool_out1_gate3 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.pool_out2_gate3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up_out1_gate0 = nn.Sequential(
            nn.ConvTranspose2d(self.nb_filter[1], self.nb_filter[0], kernel_size=2, stride=2),
            SwitchNorm2d(self.nb_filter[0], momentum=bn_momentum),
            nn.ReLU(inplace=True))
        self.up_out2_gate0 = nn.Sequential(
            nn.ConvTranspose2d(self.nb_filter[2], self.nb_filter[1], kernel_size=4, stride=4),
            SwitchNorm2d(self.nb_filter[1], momentum=bn_momentum),
            nn.ReLU(inplace=True))
        self.up_out2_gate1 = nn.Sequential(
            nn.ConvTranspose2d(self.nb_filter[2], self.nb_filter[1], kernel_size=2, stride=2),
            SwitchNorm2d(self.nb_filter[1], momentum=bn_momentum),
            nn.ReLU(inplace=True))
        self.up_out3_gate0 = nn.Sequential(
            nn.ConvTranspose2d(self.nb_filter[3], self.nb_filter[2], kernel_size=8, stride=8),
            SwitchNorm2d(self.nb_filter[2], momentum=bn_momentum),
            nn.ReLU(inplace=True))
        self.up_out3_gate1 = nn.Sequential(
            nn.ConvTranspose2d(self.nb_filter[3], self.nb_filter[2], kernel_size=4, stride=4),
            SwitchNorm2d(self.nb_filter[2], momentum=bn_momentum),
            nn.ReLU(inplace=True))
        self.up_out3_gate2 = nn.Sequential(
            nn.ConvTranspose2d(self.nb_filter[3], self.nb_filter[2], kernel_size=2, stride=2),
            SwitchNorm2d(self.nb_filter[2], momentum=bn_momentum),
            nn.ReLU(inplace=True))
        self.Fint_num = [256, 128, 64, 32]
        self.att3_1 = SN_CS_Parallel_Attention_block(F_g=self.Fint_num[0], F_x=self.Fint_num[3], F_int=self.Fint_num[0])
        self.att3_2 = SN_CS_Parallel_Attention_block(F_g=self.Fint_num[0], F_x=self.Fint_num[2], F_int=self.Fint_num[0])
        self.att3_3 = SN_CS_Parallel_Attention_block(F_g=self.Fint_num[0], F_x=self.Fint_num[1], F_int=self.Fint_num[0])
        self.att3_4 = SN_CS_Parallel_Attention_block(F_g=self.Fint_num[0], F_x=self.Fint_num[0], F_int=self.Fint_num[0])
        self.att2_1 = SN_CS_Parallel_Attention_block(F_g=self.Fint_num[1], F_x=self.Fint_num[3], F_int=self.Fint_num[1])
        self.att2_2 = SN_CS_Parallel_Attention_block(F_g=self.Fint_num[1], F_x=self.Fint_num[2], F_int=self.Fint_num[1])
        self.att2_3 = SN_CS_Parallel_Attention_block(F_g=self.Fint_num[1], F_x=self.Fint_num[1], F_int=self.Fint_num[1])
        self.att2_4 = SN_CS_Parallel_Attention_block(F_g=self.Fint_num[1], F_x=self.Fint_num[1], F_int=self.Fint_num[1])
        self.att1_1 = SN_CS_Parallel_Attention_block(F_g=self.Fint_num[2], F_x=self.Fint_num[3], F_int=self.Fint_num[2])
        self.att1_2 = SN_CS_Parallel_Attention_block(F_g=self.Fint_num[2], F_x=self.Fint_num[2], F_int=self.Fint_num[2])
        self.att1_3 = SN_CS_Parallel_Attention_block(F_g=self.Fint_num[2], F_x=self.Fint_num[2], F_int=self.Fint_num[2])
        self.att1_4 = SN_CS_Parallel_Attention_block(F_g=self.Fint_num[2], F_x=self.Fint_num[1], F_int=self.Fint_num[2])
        self.att0_1 = SN_CS_Parallel_Attention_block(F_g=self.Fint_num[3], F_x=self.Fint_num[3], F_int=self.Fint_num[3])
        self.att0_2 = SN_CS_Parallel_Attention_block(F_g=self.Fint_num[3], F_x=self.Fint_num[3], F_int=self.Fint_num[3])
        self.att0_3 = SN_CS_Parallel_Attention_block(F_g=self.Fint_num[3], F_x=self.Fint_num[2], F_int=self.Fint_num[3])
        self.att0_4 = SN_CS_Parallel_Attention_block(F_g=self.Fint_num[3], F_x=self.Fint_num[1], F_int=self.Fint_num[3])

    def get_last_feat_modules(self):
        return self.pred

    def forward(self, x, all_features=False, same=False):
        preblock = self.preBlock(x)
        f0 = preblock
        out_pool, indices0 = self.maxpool1(preblock)
        out1 = self.forw1(out_pool)
        f1 = out1
        out1_pool, indices1 = self.maxpool2(out1)
        out2 = self.forw2(out1_pool)
        f2 = out2
        out2_pool, indices2 = self.maxpool3(out2)
        out3 = self.forw3(out2_pool)
        f3 = out3
        out3_pool, indices3 = self.maxpool4(out3)
        out4 = self.forw4(out3_pool)
        f4 = out4
        out1_gate0 = self.up_out1_gate0(out1)
        out2_gate0 = self.up_out2_gate0(out2)
        out3_gate0 = self.up_out3_gate0(out3)
        preblock_gate1 = self.pool_preblock_gate1(preblock)
        out2_gate1 = self.up_out2_gate1(out2)
        out3_gate1 = self.up_out3_gate1(out3)
        preblock_gate2 = self.pool_preblock_gate2(preblock)
        out1_gate2 = self.pool_out1_gate2(out1)
        out3_gate2 = self.up_out3_gate2(out3)
        preblock_gate3 = self.pool_preblock_gate3(preblock)
        out1_gate3 = self.pool_out1_gate3(out1)
        out2_gate3 = self.pool_out2_gate3(out2)
        up4 = self.up4(out4)
        up4 = nn.functional.interpolate(up4, size=(int(self.original_size / 8), int(self.original_size / 8)), mode='nearest')
        deconv4 = self.deconv4(out4)
        up4_sig = up4 + deconv4
        up4 = torch.cat((up4_sig, deconv4), dim=1)
        preblock_gate3 = self.att3_1(g=up4_sig, x=preblock_gate3)
        out1_gate3 = self.att3_2(g=up4_sig, x=out1_gate3)
        out2_gate3 = self.att3_3(g=up4_sig, x=out2_gate3)
        out3 = self.att3_4(g=up4_sig, x=out3)
        gate3 = torch.cat((up4, preblock_gate3, out1_gate3, out2_gate3, out3), dim=1)
        comb3 = self.back1(gate3)
        f5 = comb3
        up3 = self.up3(comb3)
        up3 = nn.functional.interpolate(up3, size=(int(self.original_size / 4), int(self.original_size / 4)), mode='nearest')
        deconv3 = self.deconv3(comb3)
        up3_sig = up3 + deconv3
        up3 = torch.cat((up3, deconv3), dim=1)
        preblock_gate2 = self.att2_1(g=up3_sig, x=preblock_gate2)
        out1_gate2 = self.att2_2(g=up3_sig, x=out1_gate2)
        out2 = self.att2_3(g=up3_sig, x=out2)
        out3_gate2 = self.att2_4(g=up3_sig, x=out3_gate2)
        gate2 = torch.cat((up3, preblock_gate2, out1_gate2, out2, out3_gate2), dim=1)
        comb2 = self.back2(gate2)
        f6 = comb2
        up2 = self.up2(comb2)
        up2 = nn.functional.interpolate(up2, size=(int(self.original_size / 2), int(self.original_size / 2)), mode='nearest')
        deconv2 = self.deconv2(comb2)
        up2_sig = up2 + deconv2
        up2 = torch.cat((up2, deconv2), dim=1)
        preblock_gate1 = self.att1_1(g=up2_sig, x=preblock_gate1)
        out1 = self.att1_2(g=up2_sig, x=out1)
        out2_gate1 = self.att1_3(g=up2_sig, x=out2_gate1)
        out3_gate1 = self.att1_4(g=up2_sig, x=out3_gate1)
        gate1 = torch.cat((up2, preblock_gate1, out1, out2_gate1, out3_gate1), dim=1)
        comb1 = self.back3(gate1)
        f7 = comb1
        up1 = self.up1(comb1)
        up1 = nn.functional.interpolate(up1, size=(self.original_size, self.original_size), mode='nearest')
        deconv1 = self.deconv1(comb1)
        up1_sig = up1 + deconv1
        up1 = torch.cat((up1, deconv1), dim=1)
        preblock = self.att0_1(g=up1_sig, x=preblock)
        out1_gate0 = self.att0_2(g=up1_sig, x=out1_gate0)
        out2_gate0 = self.att0_3(g=up1_sig, x=out2_gate0)
        out3_gate0 = self.att0_4(g=up1_sig, x=out3_gate0)
        gate0 = torch.cat((up1, preblock, out1_gate0, out2_gate0, out3_gate0), dim=1)
        out = self.back4(gate0)
        feature = out
        f8 = out
        out = self.pred(out)
        out3 = self.out_conv1(comb3)
        out2 = self.out_conv2(comb2)
        out1 = self.out_conv3(comb1)
        if same:
            return out.permute(0, 2, 3, 1), out3.permute(0, 2, 3, 1), \
                   out2.permute(0, 2, 3, 1), out1.permute(0, 2, 3, 1), [f0, f1, f2, f3, f4, f5, f6, f7, f8]
        if all_features:
            return [f0, f1, f2, f3, f4, f5, f6, f7, f8], out.permute(0, 2, 3, 1)
        return out.permute(0, 2, 3, 1), out3.permute(0, 2, 3, 1), \
               out2.permute(0, 2, 3, 1), out1.permute(0, 2, 3, 1)
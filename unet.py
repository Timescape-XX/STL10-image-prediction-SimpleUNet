import torch
import torch.nn as nn
import math

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, num_classes=None):
        super(ResidualBlock, self).__init__()
        
        # time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, out_channels),
            nn.SiLU()
        )

        # label embedding
        self.label_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, out_channels),
            nn.SiLU()
        )

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)

        self.group_norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.relu = nn.ReLU()

        # channel change
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, time_emb, label_emb=None):
        # residual
        residual = self.residual_conv(x)

        # time embedding
        time_emb = self.time_mlp(time_emb)

        # label embedding
        if label_emb is not None and hasattr(self, 'label_mlp'):
            label_emb = self.label_mlp(label_emb)

        # first convolution
        h = self.conv1(x)
        h = self.group_norm(h)
        h = self.relu(h)

        # add time embedding
        h = h + time_emb.unsqueeze(2).unsqueeze(3)

        # add label embedding
        h = h + label_emb.unsqueeze(2).unsqueeze(3) if label_emb is not None else h

        # second convolution
        h = self.conv2(h)
        h = self.group_norm(h)

        # residual connection
        h += residual
        h = self.relu(h)

        return h
    
class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.group_norm=nn.GroupNorm(num_groups=8,num_channels=in_channels)
        self.q=nn.Conv2d(in_channels,in_channels,kernel_size=1) # query
        self.k=nn.Conv2d(in_channels,in_channels,kernel_size=1) # key
        self.v=nn.Conv2d(in_channels,in_channels,kernel_size=1) # value
        self.proj=nn.Conv2d(in_channels,in_channels,kernel_size=1)
    
    def forward(self,x):
        B,C,H,W=x.shape
        h=self.group_norm(x)
        q=self.q(h).view(B,C,-1).permute(0,2,1)  # B,HW,C
        k=self.k(h).view(B,C,-1)
        v=self.v(h).view(B,C,-1).permute(0,2,1)  # B,HW,C

        attn=torch.bmm(q,k)*(C**-0.5)
        attn=torch.softmax(attn,dim=-1)

        out=torch.bmm(attn,v).permute(0,2,1).view(B,C,H,W)
        return x+self.proj(out)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=128, time_emb_dim=128, num_classes=10):
        super(UNet, self).__init__()

        # time embedding dim
        self.time_emb_dim = time_emb_dim
        # sin-cos position encoding
        self.time_emb = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # label embedding
        self.label_emb = nn.Embedding(num_classes, time_emb_dim)
        self.label_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # encoder
        self.enc1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.res1 = ResidualBlock(in_channels=out_channels, out_channels=out_channels,time_emb_dim=time_emb_dim, num_classes=num_classes)
        self.down1 = nn.MaxPool2d(kernel_size=2)

        self.res2 = ResidualBlock(in_channels=out_channels, out_channels=out_channels*2, time_emb_dim=time_emb_dim, num_classes=num_classes)
        self.down2 = nn.MaxPool2d(kernel_size=2)

        # bottleneck
        self.bottleneck_res = ResidualBlock(in_channels=out_channels*2, out_channels=out_channels*2, time_emb_dim=time_emb_dim, num_classes=num_classes)
        self.bottleneck_attn = AttentionBlock(in_channels=out_channels*2)

        # drop out
        self.dropout = nn.Dropout(0.2)

        # decoder
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")  # nearest mode is more suitable for images
        self.dec1 = ResidualBlock(in_channels=out_channels*4, out_channels=out_channels*2, time_emb_dim=time_emb_dim, num_classes=num_classes)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.out_res = ResidualBlock(in_channels=out_channels*3, out_channels=out_channels, time_emb_dim=time_emb_dim, num_classes=num_classes)

        # output conv
        self.out_conv = nn.Conv2d(in_channels=out_channels, out_channels=in_channels, kernel_size=1)

    def get_time_embedding(self, t):

        t = t.float()
        half_dim = self.time_emb_dim // 2
        
        # compute positional embeddings
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        
        # concatenate sin and cos components
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb

    def forward(self, x, t, labels):
        # get time embedding
        time_emb = self.get_time_embedding(t)
        time_emb = self.time_emb(time_emb)

        # get label embedding
        label_emb = self.label_mlp(self.label_emb(labels))

        # encoder
        h1 = self.enc1(x) # [B, 128, 96, 96]
        h1 = self.res1(h1, time_emb, label_emb)  # [B, 128, 96, 96]
        s = self.down1(h1)  # [B, 128, 48, 48]

        h2 = self.res2(s, time_emb, label_emb)  # [B, 256, 48, 48]
        s = self.down2(h2)  # [B, 256, 24, 24]

        # bottleneck
        s = self.bottleneck_res(s, time_emb, label_emb)   # [B, 256, 24, 24]
        s = self.bottleneck_attn(s)

        # drop out
        s = self.dropout(s)

        # decoder
        s = self.up1(s) # [B, 256, 48, 48]
        s = torch.cat([s, h2],dim=1) # [B, 256+256, 48, 48]
        s = self.dec1(s, time_emb, label_emb)  # [B, 256, 48, 48]

        s = self.up2(s) # [B, 256, 96, 96]
        s = torch.cat([s, h1], dim=1) #[B, 256+128, 96, 96]
        s = self.out_res(s, time_emb, label_emb)  # [B, 128, 96, 96]

        s = self.out_conv(s)  # [B, in_channels, 96, 96]

        return s
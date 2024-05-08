import torch
import torch.nn as nn

from einops import rearrange
import torch.nn.functional as F

# Configuration flags and hyperparameters
USE_MAMBA = 1
DIFFERENT_H_STATES_RECURRENT_UPDATE_MECHANISM = 0
batch_size = 16
state_size = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5,
                 device: str = 'cuda'):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class PEG(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.proj = nn.Conv2d(in_c, in_c, 3, 1, 3//2, groups=in_c)
    def forward(self, x, H, W):
        B, N, C = x.shape
        cnn_in = x.transpose(1, 2).view(B, C, H, W)
        y = self.proj(cnn_in).flatten(2).transpose(1, 2)
        x = x+y

        return x

class S6(nn.Module):
    def __init__(self, seq_len, d_model, state_size, device):
        super(S6, self).__init__()

        self.fc1 = nn.Linear(d_model, d_model, device=device)
        self.fc2 = nn.Linear(d_model, state_size, device=device)
        self.fc3 = nn.Linear(d_model, state_size, device=device)

        self.seq_len = seq_len
        self.d_model = d_model
        self.state_size = state_size

        self.A = nn.Parameter(F.normalize(torch.ones(d_model, state_size, device=device), p=2, dim=-1))
        nn.init.xavier_uniform_(self.A)

        self.B = torch.zeros(batch_size, self.seq_len, self.state_size, device=device)
        self.C = torch.zeros(batch_size, self.seq_len, self.state_size, device=device)

        self.delta = torch.zeros(batch_size, self.seq_len, self.d_model, device=device)
        self.dA = torch.zeros(batch_size, self.seq_len, self.d_model, self.state_size, device=device)
        self.dB = torch.zeros(batch_size, self.seq_len, self.d_model, self.state_size, device=device)

        # h  [batch_size, seq_len, d_model, state_size]
        self.h = torch.zeros(batch_size, self.seq_len, self.d_model, self.state_size, device=device)
        self.y = torch.zeros(batch_size, self.seq_len, self.d_model, device=device)

    def discretization(self):

        self.dB = torch.einsum("bld,bln->bldn", self.delta, self.B)

        self.dA = torch.exp(torch.einsum("bld,dn->bldn", self.delta, self.A))

        return self.dA, self.dB

    def forward(self, x):

        # Algorithm 2  MAMBA paper
        self.B = self.fc2(x)
        self.C = self.fc3(x)
        self.delta = F.softplus(self.fc1(x))

        self.discretization()

        if DIFFERENT_H_STATES_RECURRENT_UPDATE_MECHANISM:

            global current_batch_size
            current_batch_size = x.shape[0]

            if self.h.shape[0] != current_batch_size:
                different_batch_size = True

                h_new = torch.einsum('bldn,bldn->bldn', self.dA, self.h[:current_batch_size, ...]) + rearrange(x,
                                                                                                   "b l d -> b l d 1") * self.dB

            else:
                different_batch_size = False
                h_new = torch.einsum('bldn,bldn->bldn', self.dA, self.h) + rearrange(x, "b l d -> b l d 1") * self.dB

                # y  [batch_size, seq_len, d_model]
                self.y = torch.einsum('bln,bldn->bld', self.C, h_new)

                global temp_buffer

                temp_buffer = h_new.detach().clone() if not self.h.requires_grad else h_new.clone()

                return self.y
        else:

            # h [batch_size, seq_len, d_model, state_size]
            h = torch.zeros(x.size(0), self.seq_len, self.d_model, self.state_size, device=x.device)
            y = torch.zeros_like(x)

            h = torch.einsum('bldn,bldn->bldn', self.dA, h) + rearrange(x, "b l d -> b l d 1") * self.dB

            # y  [batch_size, seq_len, d_model]
            y = torch.einsum('bln,bldn->bld', self.C, h)

            return y

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()

        self.groups = groups

    def forward(self, x):
        batch_size, length, channels = x.size()
        channels_per_group = channels // self.groups

        # reshape
        x = x.view(batch_size, length, self.groups, channels_per_group)
        x = torch.transpose(x, 2, 3).contiguous()

        # flatten
        x = x.view(batch_size, length, -1)
        return x

class Group_SSM(nn.Module):
    def __init__(self, group=4, sl=4096, channel=16):
        super().__init__()
        self.norm = RMSNorm(channel)
        self.pe = PEG(channel)

        self.inp_proj = nn.Linear(channel, 2 * channel, device=device)
        self.conv = nn.Conv1d(sl, sl, kernel_size=3, padding=1, device=device)

        self.S6 = S6(seq_len=sl, d_model=2*channel//group, state_size=state_size, device=device)
        self.act = nn.SiLU()
        self.shuffle = ChannelShuffle(groups=group)

        self.out_proj = nn.Linear(2 * channel, channel, device=device)

    def forward(self, input):
        b, c, h, w = input.shape
        x = input.flatten(2).permute(0,2,1)
        x = self.norm(self.pe(x, h, w))

        x = self.inp_proj(x)
        x = self.act(self.conv(x))


        x1, x2, x3, x4 = x.chunk(4, dim=-1)

        x1 = self.act(self.S6(x1))
        x2 = self.act(self.S6(x2))
        x3 = self.act(self.S6(x3))
        x4 = self.act(self.S6(x4))
        x = torch.cat([x1, x2, x3, x4], dim=-1)
        x = self.shuffle(x)
        x = self.out_proj(x)

        x = x.permute(0,2,1).view([b,c,h,w])+input

        return x


class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileMamba_blck(nn.Module):
    def __init__(self, sl, inc, d, outc):
        super().__init__()
        self.conv1 = conv_nxn_bn(inc, inc, 3)
        self.conv2 = conv_1x1_bn(inc, d)

        self.ssm = Group_SSM(sl=sl, channel=d)

        self.conv3 = conv_1x1_bn(d, outc)
        self.conv4 = conv_nxn_bn(outc, outc, 3)

    def forward(self, x):
        x = self.conv2(self.conv1(x))

        x = self.ssm(x)

        x = self.conv4(self.conv3(x))

        return x

class Mobilemamba(nn.Module):
    def __init__(self, image_size, channels, num_classes, expansion=4):
        super().__init__()
        ih, iw = image_size

        self.conv1 = conv_nxn_bn(3, channels[0], stride=2)

        self.mv2 = nn.ModuleList([])
        self.mv2.append(MV2Block(channels[0], channels[1], 1, expansion))
        self.mv2.append(MV2Block(channels[1], channels[2], 2, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))  # Repeat

        self.mv2.append(MV2Block(channels[3], channels[4], 2, expansion))
        self.mv2.append(MV2Block(channels[5], channels[6], 2, expansion))
        self.mv2.append(MV2Block(channels[7], channels[8], 2, expansion))

        self.mamba = nn.ModuleList([])
        self.mamba.append(MobileMamba_blck(sl=(ih//8)**2, d=8, inc=channels[4], outc=channels[5]))
        self.mamba.append(MobileMamba_blck(sl=(ih//16)**2, d=8, inc=channels[6], outc=channels[7]))
        self.mamba.append(MobileMamba_blck(sl=(ih//32)**2, d=8, inc=channels[8], outc=channels[9]))

        self.conv2 = conv_1x1_bn(channels[-2], channels[-1])

        self.pool = nn.AvgPool2d(ih // 32, 1)
        self.fc = nn.Linear(channels[-1], num_classes, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.mv2[0](x)

        x = self.mv2[1](x)
        x = self.mv2[2](x)
        x = self.mv2[3](x)  # Repeat

        x = self.mv2[4](x)
        x = self.mamba[0](x)

        x = self.mv2[5](x)
        x = self.mamba[1](x)

        x = self.mv2[6](x)
        x = self.mamba[2](x)
        x = self.conv2(x)

        x = self.pool(x).view(-1, x.shape[1])
        x = self.fc(x)
        return x


def mobilemamba_xxs():
    channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    return Mobilemamba((256, 256), channels, num_classes=3, expansion=2)


def mobilemamba_xs():
    channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
    return Mobilemamba((256, 256), channels, num_classes=3)


def mobilemamba_s():
    channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
    return Mobilemamba((256, 256), channels, num_classes=3)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    img = torch.randn(5, 3, 256, 256)

    vit = mobilemamba_xxs()
    vit.to(device)
    out = vit(img.to(device))
    print(out.shape)
    print(count_parameters(vit))

    #vit = mobilevit_xs()
    #out = vit(img)
    #print(out.shape)
    #print(count_parameters(vit))

    #vit = mobilevit_s()
    #out = vit(img)
    #print(out.shape)
    #print(count_parameters(vit))

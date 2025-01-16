# -*- coding: utf-8 -*-
"""NeRF2 NN model
"""
from tokenize import Double, Double3
from pyparsing import C
from sklearn.calibration import cross_val_predict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from convkan import ConvKAN, LayerNorm2D

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
img2me = lambda x, y : torch.mean(abs(x - y))
sig2mse = lambda x, y : torch.mean((x - y) ** 2)
csi2snr = lambda x, y: -10 * torch.log10(
    torch.norm(x - y, dim=(1, 2)) ** 2 /
    torch.norm(y, dim=(1, 2)) ** 2
)

def loss_function(recon_x, x, mu, logvar):
    # Binary Cross-Entropy Loss
    BCE = nn.functional.binary_cross_entropy_with_logits(recon_x, x, reduction='sum')
    # Kullback-Leibler Divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi
    
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    

class PyramidPooling(nn.Module):
    def __init__(self, in_channels, pool_sizes):
        super(PyramidPooling, self).__init__()
        self.pools = nn.ModuleList([nn.AdaptiveAvgPool2d(output_size=(size, size)) for size in pool_sizes])
        self.convs = nn.ModuleList([nn.Conv2d(in_channels, in_channels // len(pool_sizes), kernel_size=1) for _ in pool_sizes])
        self.out_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, x):
        features = [x]
        for pool, conv in zip(self.pools, self.convs):
            p = pool(x)
            p = conv(p)
            p = F.interpolate(p, size=x.shape[2:], mode='bilinear', align_corners=False)
            features.append(p)
        x = torch.cat(features, dim=1)
        x = self.out_conv(x)
        return x
    


class UNet2(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=64, pool_size=[1,2]):
        super(UNet2, self).__init__()
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = DoubleConv(base_channels, 2*base_channels)
        self.down2 = DoubleConv(2*base_channels, 4*base_channels)
        self.down3 = DoubleConv(4*base_channels, 8*base_channels)
        
        self.up1 = nn.ConvTranspose2d(8*base_channels, 4*base_channels, kernel_size=2, stride=2)
        self.att1 = AttentionGate(4*base_channels, 4*base_channels, 2*base_channels)
        self.upconv1 = DoubleConv(8*base_channels, 4*base_channels)
        
        self.up2 = nn.ConvTranspose2d(4*base_channels, 2*base_channels, kernel_size=2, stride=2)
        self.att2 = AttentionGate(2*base_channels, 2*base_channels, base_channels)
        self.upconv2 = DoubleConv(4*base_channels, 2*base_channels)
        
        self.up3 = nn.ConvTranspose2d(2*base_channels, base_channels, kernel_size=2, stride=2)
        self.att3 = AttentionGate(base_channels, base_channels, base_channels//2)
        self.upconv3 = DoubleConv(2*base_channels, base_channels)

        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        self.ppm = PyramidPooling(4*base_channels, pool_size)

    def adaptive_pooling(self, x):
        _, _, h, w = x.size()
        pad_h = h % 2
        pad_w = w % 2
        if pad_h != 0 or pad_w != 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
        return F.max_pool2d(x, kernel_size=2, stride=2)
    
    def crop_to_match(self, x1, x2):
        if x1.size(2) > x2.size(2):
            x1 = x1[:, :, :x2.size(2)]
        elif x2.size(2) > x1.size(2):
            x2 = x2[:, :, :x1.size(2)]
        return x1, x2
    



    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.adaptive_pooling(x1)
        x2 = self.down1(x2)
        x3 = self.adaptive_pooling(x2)
        x3 = self.down2(x3)
        # x4 = self.adaptive_pooling(x3)
        # x4 = self.down3(x4)

        x3 = self.ppm(x3)

        # x = self.up1(x4)

        # x, x3 = self.crop_to_match(x, x3)

        # x = self.att1(x, x3)
        # x = torch.cat([x, x3], dim=1)
        # x = self.upconv1(x)

        x = self.up2(x3)

        x, x2 = self.crop_to_match(x, x2)
        x = self.att2(x, x2)
        x = torch.cat([x, x2], dim=1)
        x = self.upconv2(x)

        x = self.up3(x)

        x, x1 = self.crop_to_match(x, x1)
        x = self.att3(x, x1)
        x = torch.cat([x, x1], dim=1)
        x = self.upconv3(x)

        x = self.outc(x)
        return x.permute(0, 2, 3, 1) 
    


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=64, pool_size=[1,2]):
        super(UNet, self).__init__()
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = DoubleConv(base_channels, 2*base_channels)
        self.down2 = DoubleConv(2*base_channels, 4*base_channels)
        self.down3 = DoubleConv(4*base_channels, 8*base_channels)
        
        self.up1 = nn.ConvTranspose2d(8*base_channels, 4*base_channels, kernel_size=2, stride=2)
        self.att1 = AttentionGate(4*base_channels, 4*base_channels, 2*base_channels)
        self.upconv1 = DoubleConv(8*base_channels, 4*base_channels)
        
        self.up2 = nn.ConvTranspose2d(4*base_channels, 2*base_channels, kernel_size=2, stride=2)
        self.att2 = AttentionGate(2*base_channels, 2*base_channels, base_channels)
        self.upconv2 = DoubleConv(4*base_channels, 2*base_channels)
        
        self.up3 = nn.ConvTranspose2d(2*base_channels, base_channels, kernel_size=2, stride=2)
        self.att3 = AttentionGate(base_channels, base_channels, base_channels//2)
        self.upconv3 = DoubleConv(2*base_channels, base_channels)

        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        self.ppm = PyramidPooling(8*base_channels, pool_size)

    def adaptive_pooling(self, x):
        _, _, h, w = x.size()
        pad_h = h % 2
        pad_w = w % 2
        if pad_h != 0 or pad_w != 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
        return F.max_pool2d(x, kernel_size=2, stride=2)
    
    def crop_to_match(self, x1, x2):
        if x1.size(2) > x2.size(2):
            x1 = x1[:, :, :x2.size(2)]
        elif x2.size(2) > x1.size(2):
            x2 = x2[:, :, :x1.size(2)]
        return x1, x2
    



    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.adaptive_pooling(x1)
        x2 = self.down1(x2)
        x3 = self.adaptive_pooling(x2)
        x3 = self.down2(x3)
        x4 = self.adaptive_pooling(x3)
        x4 = self.down3(x4)

        x4 = self.ppm(x4)

        x = self.up1(x4)

        x, x3 = self.crop_to_match(x, x3)

        x = self.att1(x, x3)
        x = torch.cat([x, x3], dim=1)
        x = self.upconv1(x)

        x = self.up2(x3)

        x, x2 = self.crop_to_match(x, x2)
        x = self.att2(x, x2)
        x = torch.cat([x, x2], dim=1)
        x = self.upconv2(x)

        x = self.up3(x)

        x, x1 = self.crop_to_match(x, x1)
        x = self.att3(x, x1)
        x = torch.cat([x, x1], dim=1)
        x = self.upconv3(x)

        x = self.outc(x)
        return x.permute(0, 2, 3, 1) 



class Embedder():
    """positional encoding
    """
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']    # input dimension of gamma
        out_dim = 0

        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']    # L-1, 10-1 by default
        N_freqs = self.kwargs['num_freqs']         # L


        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)  #2^[0,1,...,L-1]
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq) + x)  # PE
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        """return: gamma(input)
        """
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)




def get_embedder(multires, is_embeded=True, input_dims=3):
    """get positional encoding function

    Parameters
    ----------
    multires : log2 of max freq for positional encoding, i.e., (L-1)
    i : set 1 for default positional encoding, 0 for none
    input_dims : input dimension of gamma


    Returns
    -------
        embedding function; output_dims
    """
    if is_embeded == False:
        return nn.Identity(), input_dims

    embed_kwargs = {
                'include_input' : False,
                'input_dims' : input_dims,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim  # PE


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(MLP, self).__init__()
        
        # 定义第一层
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.dropout1 = nn.Dropout(0.25)
        
        # 定义第二层
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.dropout2 = nn.Dropout(0.25)
        
        # 定义第三层
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.bn3 = nn.BatchNorm1d(hidden_size3)
        self.dropout3 = nn.Dropout(0.25)

        self.fc4 = nn.Linear(hidden_size3, hidden_size3)
        self.bn4 = nn.BatchNorm1d(hidden_size3)
        self.dropout4 = nn.Dropout(0.25)
        
        # 定义输出层
        self.fc5 = nn.Linear(hidden_size3, output_size)

    def forward(self, x):
        # 通过第一层
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        # 通过第二层
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        # 通过第三层
        x = F.leaky_relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)

        x = F.leaky_relu(self.bn4(self.fc4(x)))
        x = self.dropout4(x)
        
        # 输出层
        x = self.fc5(x)
        return x


class DoubleConv1D(nn.Module):
    """(convolution => [BN] => ReLU) * 2 in 1D"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv1D, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class AttentionGate1D(nn.Module):
    """ Attention Gate for 1D data """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate1D, self).__init__()
        self.W_g = nn.Sequential(nn.Conv1d(F_g, F_int, kernel_size=1, bias=False),
                                 nn.BatchNorm1d(F_int))
        self.W_x = nn.Sequential(nn.Conv1d(F_l, F_int, kernel_size=1, bias=False),
                                 nn.BatchNorm1d(F_int))
        self.psi = nn.Sequential(nn.Conv1d(F_int, 1, kernel_size=1, bias=False),
                                 nn.BatchNorm1d(1),
                                 nn.Sigmoid())

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = F.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class UNet1D(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=64):
        super(UNet1D, self).__init__()
        self.inc = DoubleConv1D(in_channels, base_channels)
        self.down1 = DoubleConv1D(base_channels, 2*base_channels)
        self.down2 = DoubleConv1D(2*base_channels, 4*base_channels)
        self.down3 = DoubleConv1D(4*base_channels, 8*base_channels)
        
        self.up1 = nn.ConvTranspose1d(8*base_channels, 4*base_channels, kernel_size=2, stride=2)
        self.att1 = AttentionGate1D(4*base_channels, 4*base_channels, 2*base_channels)
        self.upconv1 = DoubleConv1D(8*base_channels, 4*base_channels)
        
        self.up2 = nn.ConvTranspose1d(4*base_channels, 2*base_channels, kernel_size=2, stride=2)
        self.att2 = AttentionGate1D(2*base_channels, 2*base_channels, base_channels)
        self.upconv2 = DoubleConv1D(4*base_channels, 2*base_channels)
        
        self.up3 = nn.ConvTranspose1d(2*base_channels, base_channels, kernel_size=2, stride=2)
        self.att3 = AttentionGate1D(base_channels, base_channels, base_channels//2)
        self.upconv3 = DoubleConv1D(2*base_channels, base_channels)

        self.outc = nn.Conv1d(base_channels, out_channels, kernel_size=1)

    def crop_to_match(self, x1, x2):
        if x1.size(2) > x2.size(2):
            x1 = x1[:, :, :x2.size(2)]
        elif x2.size(2) > x1.size(2):
            x2 = x2[:, :, :x1.size(2)]
        return x1, x2

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4)
        x, x3 = self.crop_to_match(x, x3)
        x = self.att1(x, x3)
        x = torch.cat([x, x3], dim=1)
        x = self.upconv1(x)

        x = self.up2(x)
        x, x2 = self.crop_to_match(x, x2)
        x = self.att2(x, x2)
        x = torch.cat([x, x2], dim=1)
        x = self.upconv2(x)

        x = self.up3(x)
        x, x1 = self.crop_to_match(x, x1)
        x = self.att3(x, x1)
        x = torch.cat([x, x1], dim=1)
        x = self.upconv3(x)

        x = self.outc(x)
        return x.permute(0, 2, 1)
    




class NeRF2(nn.Module):

    def __init__(self, D=8, W=256, skips=[4],
                 input_dims={'pts':3, 'view':3, 'tx':3},
                 multires = {'pts':10, 'view':10, 'tx':10},
                 is_embeded={'pts':True, 'view':True, 'tx':False},
                 attn_output_dims=2, sig_output_dims=2, n_samples=64, m='PPM'
                ):
        """NeRF2 model

        Parameters
        ----------
        D : int, hidden layer number, default by 8
        W : int, Dimension per hidden layer, default by 256
        skip : list, skip layer index
        input_dims: dict, input dimensions
        multires: dict, log2 of max freq for position, view, and tx position positional encoding, i.e., (L-1)
        is_embeded : dict, whether to use positional encoding
        attn_output_dims : int, output dimension of attenuation
        sig_output_dims : int, output dimension of signal
        """

        super().__init__()
        self.skips = skips

        # set positional encoding function
        self.embed_pts_fn, input_pts_dim = get_embedder(multires['pts'], is_embeded['pts'], input_dims['pts'])
        self.embed_view_fn, input_view_dim = get_embedder(multires['view'], is_embeded['view'], input_dims['view'])
        self.embed_tx_fn, input_tx_dim = get_embedder(multires['tx'], is_embeded['tx'], input_dims['tx'])
        self.m = m

        ## attenuation network
        self.attenuation_linears = nn.ModuleList(
            [nn.Linear(input_pts_dim, W)] +
            [nn.Linear(W, W) if i not in skips else nn.Linear(W + input_pts_dim, W)
             for i in range(D - 1)]
        )

        if self.m == 'PPM':
            self.attenuation_U_layer = nn.ModuleList(
                [UNet(input_pts_dim, W)] + 
                [nn.Linear(W, W)]
        )
            self.signal_U_layer = nn.ModuleList(
                [UNet(input_view_dim + input_tx_dim + W, W//2)] +
                [nn.Linear(W//2, W//2)]
        )
            self.attenuation_U_layer1d = nn.ModuleList(
                [UNet1D(input_pts_dim, W)] + 
                [nn.Linear(W, W)]
        )
            self.signal_U_layer1d = nn.ModuleList(
                [UNet1D(input_view_dim + input_tx_dim + W, W//2)] +
                [nn.Linear(W//2, W//2)]
        )
            
        elif self.m == 'unet1':
            self.attenuation_U_1layer = nn.ModuleList(
                [SimpleUNet(input_pts_dim, W)] + 
                [nn.Linear(W, W)]
        )
            self.signal_U_1layer = nn.ModuleList(
                [SimpleUNet(input_view_dim + input_tx_dim + W, W//2)] +
                [nn.Linear(W//2, W//2)]
        )   
        elif self.m == 'unet3':
            self.attenuation_U_3layer = nn.ModuleList(
                [UNet3Layer(input_pts_dim, W)] + 
                [nn.Linear(W, W)]
        )
            self.signal_U_3layer = nn.ModuleList(
                [UNet3Layer(input_view_dim + input_tx_dim + W, W//2)] +
                [nn.Linear(W//2, W//2)]
        )
        
        elif self.m == 'unet2':
            self.attenuation_U_2layer = nn.ModuleList(
                [UNet2Down(input_pts_dim, W)] + 
                [nn.Linear(W, W)]
        )
            self.signal_U_2layer = nn.ModuleList(
                [UNet2Down(input_view_dim + input_tx_dim + W, W//2)] +
                [nn.Linear(W//2, W//2)]
        )
        
        elif self.m == 'unet2f':
            self.attenuation_U_layer2 = nn.ModuleList(
                [UNet2(input_pts_dim, W)] + 
                [nn.Linear(W, W)]
        )
            self.signal_U_layer2 = nn.ModuleList(
                [UNet2(input_view_dim + input_tx_dim + W, W//2)] +
                [nn.Linear(W//2, W//2)]
        )




        ## signal network
        self.signal_linears = nn.ModuleList(
            [nn.Linear(input_view_dim + input_tx_dim + W, W)] +
            [nn.Linear(W, W//2)]
        )
        


        ## output head, 2 for amplitude and phase        
        self.attenuation_output = nn.Linear(W, attn_output_dims)
        self.feature_layer = nn.Linear(W, W)
        self.signal_output = nn.Linear(W//2, sig_output_dims)


    def forward(self, pts, view, tx):
        """forward function of the model

        Parameters
        ----------
        pts: [batchsize, n_samples, 3], position of voxels
        view: [batchsize, n_samples, 3], view direction
        tx: [batchsize, n_samples, 3], position of transmitter

        Returns
        ----------
        outputs: [batchsize, n_samples, 4].   attn_amp, attn_phase, signal_amp, signal_phase
        """

        # position encoding
        pts = self.embed_pts_fn(pts).contiguous()
        view = self.embed_view_fn(view).contiguous()
        tx = self.embed_tx_fn(tx).contiguous()
        shape = pts.shape

        # if Linear
        # pts = pts.view(-1, list(pts.shape)[-1])
        # view = view.view(-1, list(view.shape)[-1])
        # tx = tx.view(-1, list(tx.shape)[-1])
        x = pts
        
        # # if Unet
        if len(x.shape) != 3 and self.m == 'PPM':
            if len(x.shape) != 3:
                x = pts.permute(0, 3, 1, 2)
        
                for i, layer in enumerate(self.attenuation_U_layer):
                    if i == 0:
                        x = layer(x)
                    else:
                        x = F.relu(layer(x))
            
            else:
                x = pts.permute(0, 2, 1)
        
                for i, layer in enumerate(self.attenuation_U_layer1d):
                    if i == 0:
                        x = layer(x)
                    else:
                        x = F.relu(layer(x))

        elif self.m == 'unet3':
            x = pts.permute(0, 3, 1, 2)
        
            for i, layer in enumerate(self.attenuation_U_3layer):
                if i == 0:
                    x = layer(x)
                else:
                    x = F.relu(layer(x))

        elif self.m == 'unet1':
            x = pts.permute(0, 3, 1, 2)
        
            for i, layer in enumerate(self.attenuation_U_1layer):
                if i == 0:
                    x = layer(x)
                else:
                    x = F.relu(layer(x))
            
        elif self.m == 'unet2':
            x = pts.permute(0, 3, 1, 2)
        
            for i, layer in enumerate(self.attenuation_U_2layer):
                if i == 0:
                    x = layer(x)
                else:
                    x = F.relu(layer(x))

        elif self.m == 'unet2f':
            x = pts.permute(0, 3, 1, 2)
        
            for i, layer in enumerate(self.attenuation_U_layer2):
                if i == 0:
                    x = layer(x)
                else:
                    x = F.relu(layer(x))
            

        # if Transformer_VAE
        # x = pts
        # for i, layer in enumerate(self.attenuation_T_layer):
        #     if i == 0:
        #         x, a_mu, a_std = layer(x)
        #         x = F.leaky_relu(x[0])
        #     else:
        #         x = F.leaky_relu(layer(x))


        # if Linear
        # for i, layer in enumerate(self.attenuation_linears):
        #     x = F.relu(layer(x))
        #     if i in self.skips:
        #         x = torch.cat([pts, x], -1)

        attn = self.attenuation_output(x)    # (batch_size, 2)
        feature = self.feature_layer(x)
        x = torch.cat([feature, view, tx], -1) 


        # if Unet
        if len(x.shape) != 3 and self.m == 'PPM':
            if len(x.shape) != 3:
                x = x.permute(0, 3, 1, 2)
        
                for i, layer in enumerate(self.signal_U_layer):
                    if i == 0:
                        x = layer(x)
                    else:
                        x = F.relu(layer(x))
            
            else:
                x = x.permute(0, 2, 1)
        
                for i, layer in enumerate(self.signal_U_layer1d):
                    if i == 0:
                        x = layer(x)
                    else:
                        x = F.relu(layer(x))

        elif self.m == 'unet3':
            x = x.permute(0, 3, 1, 2)
        
            for i, layer in enumerate(self.signal_U_3layer):
                if i == 0:
                    x = layer(x)
                else:
                    x = F.relu(layer(x))

        elif self.m == 'unet1':
            x = x.permute(0, 3, 1, 2)
        
            for i, layer in enumerate(self.signal_U_1layer):
                if i == 0:
                    x = layer(x)
                else:
                    x = F.relu(layer(x))
            
        elif self.m == 'unet2':
            x = x.permute(0, 3, 1, 2)
        
            for i, layer in enumerate(self.signal_U_2layer):
                if i == 0:
                    x = layer(x)
                else:
                    x = F.relu(layer(x))

        elif self.m == 'unet2f':
            x = x.permute(0, 3, 1, 2)
        
            for i, layer in enumerate(self.signal_U_layer2):
                if i == 0:
                    x = layer(x)
                else:
                    x = F.relu(layer(x))


        

        # if Linear
        # for i, layer in enumerate(self.signal_linears):
        #     x = F.relu(layer(x))

        # if Transformer_VAE
        # for i, layer in enumerate(self.signal_T_layer):
        #     if i == 0:
        #         x, s_mu, s_std = layer(x)
        #         x = F.leaky_relu(x[0])
        #     else:
        #         x = F.leaky_relu(layer(x))

        signal = self.signal_output(x)    #[batchsize, n_samples, 2]

        outputs = torch.cat([attn, signal], -1).contiguous()    # [batchsize, n_samples, 4]

        # if U_net / Linear
        return outputs.view(shape[:-1]+outputs.shape[-1:])
        
        # if Transformer_VAE
        # return outputs.view(shape[:-1]+outputs.shape[-1:]), [a_mu, a_std], [s_mu, s_std]


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)
        self.fc4 = nn.Linear(input_dim, 1)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        out = self.fc4(x_recon)
        loss = loss_function(x_recon.detach().cpu(), x.detach().cpu(), mu, logvar)
        return out, loss


class UNet2Down(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=64):
        super(UNet2Down, self).__init__()
        
        # Encoder部分
        self.inc = DoubleConv(in_channels, base_channels)  # 不下采样的初始卷积层
        
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_channels, base_channels * 2)
        )
        
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_channels * 2, base_channels * 4)  # Bottleneck区域
        )
        
        # Decoder部分
        self.up1 = nn.ConvTranspose2d(base_channels*4, base_channels*2, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(base_channels*4, base_channels*2)  # 与down1对应的skip
        
        self.up2 = nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(base_channels*2, base_channels)    # 与inc对应的skip
        
        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)    # [B, base, H, W]
        x2 = self.down1(x1) # [B, base*2, H/2, W/2]
        x3 = self.down2(x2) # [B, base*4, H/4, W/4]

        # Decoder
        x = self.up1(x3) # [B, base*2, H/2, W/2]
        # 拼接 skip 连接
        x = torch.cat([x2, x], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x) # [B, base, H, W]
        x = torch.cat([x1, x], dim=1)
        x = self.conv2(x)

        logits = self.outc(x) # [B, out_channels, H, W]
        return logits.permute(0, 2, 3, 1) 

class UNet3Layer(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=64, pool_size=[1,2]):
        super(UNet3Layer, self).__init__()
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = DoubleConv(base_channels, 2*base_channels)
        self.down2 = DoubleConv(2*base_channels, 4*base_channels)
        self.down3 = DoubleConv(4*base_channels, 8*base_channels)
        
        self.up1 = nn.ConvTranspose2d(8*base_channels, 4*base_channels, kernel_size=2, stride=2)
        self.upconv1 = DoubleConv(8*base_channels, 4*base_channels)
        
        self.up2 = nn.ConvTranspose2d(4*base_channels, 2*base_channels, kernel_size=2, stride=2)
        self.upconv2 = DoubleConv(4*base_channels, 2*base_channels)
        
        self.up3 = nn.ConvTranspose2d(2*base_channels, base_channels, kernel_size=2, stride=2)
        self.upconv3 = DoubleConv(2*base_channels, base_channels)

        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def adaptive_pooling(self, x):
        _, _, h, w = x.size()
        pad_h = h % 2
        pad_w = w % 2
        if pad_h != 0 or pad_w != 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
        return F.max_pool2d(x, kernel_size=2, stride=2)
    
    def crop_to_match(self, x1, x2):
        if x1.size(2) > x2.size(2):
            x1 = x1[:, :, :x2.size(2)]
        elif x2.size(2) > x1.size(2):
            x2 = x2[:, :, :x1.size(2)]
        return x1, x2
    



    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.adaptive_pooling(x1)
        x2 = self.down1(x2)
        x3 = self.adaptive_pooling(x2)
        x3 = self.down2(x3)
        x4 = self.adaptive_pooling(x3)
        x4 = self.down3(x4)


        x = self.up1(x4)

        x, x3 = self.crop_to_match(x, x3)
        x = torch.cat([x, x3], dim=1)
        x = self.upconv1(x)

        x = self.up2(x3)

        x, x2 = self.crop_to_match(x, x2)
        x = torch.cat([x, x2], dim=1)
        x = self.upconv2(x)

        x = self.up3(x)

        x, x1 = self.crop_to_match(x, x1)
        x = torch.cat([x, x1], dim=1)
        x = self.upconv3(x)

        x = self.outc(x)
        return x.permute(0, 2, 3, 1) 

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=64):
        super(SimpleUNet, self).__init__()
        
        # Encoder部分
        self.inc = DoubleConv(in_channels, base_channels)  # 不下采样的初始卷积层
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_channels, base_channels * 2)
        )
        
        # Decoder部分
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(base_channels * 2, base_channels)  # 与down1对应的skip
        
        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)     # [B, base, H, W]
        x2 = self.down1(x1)  # [B, base*2, H/2, W/2]

        # Decoder
        x = self.up1(x2)     # [B, base, H, W]
        # 拼接 skip 连接
        x = torch.cat([x1, x], dim=1)
        x = self.conv1(x)

        logits = self.outc(x) # [B, out_channels, H, W]
        return logits.permute(0, 2, 3, 1)  # 改变输出维度，以适配某些后续处理

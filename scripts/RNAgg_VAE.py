import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

class ChannelSELayer(nn.Module):
    def __init__(self, num_channels, reduction=3):
        super().__init__()
        reduced = max(1, num_channels // reduction)
        self.fc1 = nn.Linear(num_channels, reduced)
        self.fc2 = nn.Linear(reduced, num_channels)

    def forward(self, x):
        b, c, h, w = x.size()
        y = x.view(b, c, -1).mean(dim=2)
        y = self.fc1(y)
        y = F.relu(y)
        y = self.fc2(y)
        y = torch.sigmoid(y).view(b, c, 1, 1)
        return x * y


class ConvEncoder(nn.Module):
    def __init__(self, in_channels, latent_dim=128, n_layers=3, input_hw=(64, 64)):
        super().__init__()
        layers = []
        in_ch = in_channels
        for i in range(n_layers):
            out_ch = in_ch * 2
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch
        self.net = nn.Sequential(*layers)

        H, W = input_hw
        for _ in range(n_layers):
            H = (H + 2 * 1 - (4 - 1) - 1) // 2 + 1
            W = (W + 2 * 1 - (4 - 1) - 1) // 2 + 1
        self.out_hw = (in_ch, H, W)
        print('encoder out_hw:', in_ch, H, W)

        enc_ch, enc_h, enc_w = self.out_hw
        enc_flat = enc_ch * enc_h * enc_w

        hid1 = min(enc_flat, 4096)
        hid2 = min(max(516, hid1 // 2), 2048)
        hid3 = 516
        self.enc_fc1 = nn.Linear(enc_flat, hid1)
        self.enc_bn1 = nn.BatchNorm1d(hid1)
        self.enc_fc2 = nn.Linear(hid1, hid2)
        self.enc_bn2 = nn.BatchNorm1d(hid2)
        self.enc_fc3 = nn.Linear(hid2, hid3)
        self.enc_bn3 = nn.BatchNorm1d(hid3)

        # mu, var
        self.fc_mu = nn.Linear(hid3, latent_dim)
        self.fc_var = nn.Linear(hid3, latent_dim)

        # store latent dim
        self.latent_dim = latent_dim

    def forward(self, x):
        # x -> conv features
        h = self.net(x)
        b = h.size(0)
        h_flat = h.view(b, -1)

        # fc encoder
        h_e = self.enc_fc1(h_flat)
        h_e = self.enc_bn1(h_e)
        h_e = F.relu(h_e)
        h_e = self.enc_fc2(h_e)
        h_e = self.enc_bn2(h_e)
        h_e = F.relu(h_e)
        h_e = self.enc_fc3(h_e)
        h_e = self.enc_bn3(h_e)
        h_e = F.relu(h_e)

        mu = self.fc_mu(h_e)
        # ensure variance is positive for reparameterization
        var = F.softplus(self.fc_var(h_e))

        # return stats (reparameterization moved to Conv_VAE)
        return mu, var


class ConvDecoder(nn.Module):
    def __init__(self, enc_ch, enc_h, enc_w, latent_dim=128, n_layers=3, reduction=3):
        super().__init__()
        self.enc_ch = enc_ch
        self.enc_h = enc_h
        self.enc_w = enc_w
        enc_flat = enc_ch * enc_h * enc_w

        # decoder FCs (kept same sizing logic)
        self.dec_fc1 = nn.Linear(latent_dim, 516)
        self.dec_bn1 = nn.BatchNorm1d(516)
        self.dec_fc2 = nn.Linear(516, min(max(516, min(enc_flat, 4096) // 2), 2048))
        self.dec_bn2 = nn.BatchNorm1d(self.dec_fc2.out_features)
        self.dec_fc3 = nn.Linear(self.dec_fc2.out_features, min(min(enc_flat, 4096), 4096))
        self.dec_bn3 = nn.BatchNorm1d(self.dec_fc3.out_features)
        self.fc_decode = nn.Linear(self.dec_fc3.out_features, enc_flat)

        # build conv-transpose trunk that consumes (enc_ch, enc_h, enc_w)
        layers = []
        in_ch = enc_ch
        H, W = enc_h, enc_w
        for i in range(n_layers):
            out_ch = max(1, in_ch // 2)
            layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch
        self.net = nn.Sequential(*layers)

        for _ in range(n_layers):
            H = (H - 1) * 2 - 2 * 1 + 4
            W = (W - 1) * 2 - 2 * 1 + 4
            in_ch = out_ch
        self.out_hw = (in_ch, H, W)
        print('decoder out_hw:', in_ch, H, W)

        # attach attention layer same as before
        dec_out_ch = enc_ch // (2 ** n_layers)
        self.attention = ChannelSELayer(dec_out_ch, reduction=reduction)

    def forward(self, z):
        h_d = self.dec_fc1(z)
        h_d = self.dec_bn1(h_d)
        h_d = F.relu(h_d)
        h_d = self.dec_fc2(h_d)
        h_d = self.dec_bn2(h_d)
        h_d = F.relu(h_d)
        h_d = self.dec_fc3(h_d)
        h_d = self.dec_bn3(h_d)
        h_d = F.relu(h_d)
        h_flat_decode = self.fc_decode(h_d)

        # reshape to conv feature shape and run conv-transpose trunk
        b = z.size(0)
        h_conv = h_flat_decode.view(b, self.enc_ch, self.enc_h, self.enc_w)
        return self.net(h_conv)


class Conv_VAE(nn.Module):
    def __init__(self, input_shape, latent_dim=128, n_layers=3, reduction=3, device='cpu'):
        super().__init__()
        self.device = device
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        C, H, W = input_shape
        self.n_layers = n_layers

        # encoder: now returns z, mu, var
        self.attention = ChannelSELayer(C, reduction=reduction)
        self.encoder = ConvEncoder(C, latent_dim=latent_dim, n_layers=n_layers, input_hw=(H, W))
        enc_ch, enc_h, enc_w = self.encoder.out_hw
        self.enc_shape = (enc_ch, enc_h, enc_w)
        print('after-encoder', self.enc_shape)

        # decoder
        self.decoder = ConvDecoder(enc_ch=enc_ch, enc_h=enc_h, enc_w=enc_w, latent_dim=latent_dim, n_layers=n_layers, reduction=reduction)

    def forward(self, x):
        x = self.attention(x)
        mu, var = self.encoder(x)
        z = self.reparameterize(mu, var)
        x_rec = torch.sigmoid(self.decoder(z))
        return x_rec, mu, var

    def reparameterize(self, mu, var):
        std = torch.sqrt(var + 1e-8)
        eps = torch.randn_like(std)
        return mu + eps * std



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sanity-check Conv_VAE')
    parser.add_argument('--input-shape', default='11,64,64', help='C,H,W for conv model (comma-separated)')
    parser.add_argument('--latent-dim', type=int, default=64)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

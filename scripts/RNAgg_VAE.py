import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelSELayer(nn.Module):
    def __init__(self, num_channels, reduction=16):
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
    def __init__(self, in_channels, n_layers=3, input_hw=(64, 64)):
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

    def forward(self, x):
        return self.net(x)


class ConvDecoder(nn.Module):
    def __init__(self, start_in_ch, n_layers=3):
        super().__init__()
        layers = []
        in_ch = start_in_ch
        for i in range(n_layers):
            out_ch = max(1, in_ch // 2)
            layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch
        self.feature_net = nn.Sequential(*layers)
        self.final_conv = None

    def forward(self, x):
        h = self.feature_net(x)
        if self.final_conv is not None:
            return self.final_conv(h)
        return h


class Conv_VAE(nn.Module):
    def __init__(self, input_shape, latent_dim=128, base_channels=32, n_layers=3, reduction=16, device='cpu'):
        super().__init__()
        self.device = device
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        C, H, W = input_shape
        self.n_layers = n_layers

        self.attention = ChannelSELayer(C, reduction=reduction)

        self.encoder = ConvEncoder(C, n_layers=n_layers, input_hw=(H, W))
        enc_ch, enc_h, enc_w = self.encoder.out_hw
        self.enc_shape = (enc_ch, enc_h, enc_w)
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

        self.fc_mu = nn.Linear(hid3, latent_dim)
        self.fc_var = nn.Linear(hid3, latent_dim)

        self.dec_fc1 = nn.Linear(latent_dim, hid3)
        self.dec_bn1 = nn.BatchNorm1d(hid3)
        self.dec_fc2 = nn.Linear(hid3, hid2)
        self.dec_bn2 = nn.BatchNorm1d(hid2)
        self.dec_fc3 = nn.Linear(hid2, hid1)
        self.dec_bn3 = nn.BatchNorm1d(hid1)

        self.fc_decode = nn.Linear(hid1, enc_flat)
        self.decoder = ConvDecoder(start_in_ch=enc_ch, n_layers=n_layers)
        self.decoder.final_conv = nn.Conv2d(max(1, enc_ch // (2 ** n_layers)), C, kernel_size=1)

    def reparameterize(self, mu, var):
        std = torch.sqrt(var + 1e-8)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        x = self.attention(x)
        h = self.encoder(x)
        b, c, h1, w1 = h.size()
        h_flat = h.view(b, -1)
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
        var = F.softplus(self.fc_var(h_e))
        return mu, var, (c, h1, w1)

    def decode(self, z, conv_shape):
        b = z.size(0)
        h_d = self.dec_fc1(z)
        h_d = self.dec_bn1(h_d)
        h_d = F.relu(h_d)
        h_d = self.dec_fc2(h_d)
        h_d = self.dec_bn2(h_d)
        h_d = F.relu(h_d)
        h_d = self.dec_fc3(h_d)
        h_d = self.dec_bn3(h_d)
        h_d = F.relu(h_d)
        h_flat = self.fc_decode(h_d)
        c, h1, w1 = conv_shape
        h = h_flat.view(b, c, h1, w1)
        x_rec = self.decoder(h)
        # If spatial size changed due to stride/rounding, resize back to original input size
        C_in, H_in, W_in = self.input_shape
        if x_rec.size(2) != H_in or x_rec.size(3) != W_in:
            x_rec = F.interpolate(x_rec, size=(H_in, W_in), mode='bilinear', align_corners=False)
        return x_rec

    def forward(self, x):
        mu, var, conv_shape = self.encode(x)
        z = self.reparameterize(mu, var)
        x_rec = self.decode(z, conv_shape)
        return x_rec, mu, var


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Sanity-check Conv_VAE')
    parser.add_argument('--input-shape', default='11,64,64', help='C,H,W for conv model (comma-separated)')
    parser.add_argument('--latent-dim', type=int, default=64)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    device = torch.device(args.device)

    def count_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    C, H, W = [int(x) for x in args.input_shape.split(',')]
    model = Conv_VAE((C, H, W), latent_dim=args.latent_dim).to(device)
    x = torch.randn(2, C, H, W).to(device)
    out, mu, var = model(x)
    print('model=Conv_VAE, params=', count_params(model))
    print('input', x.shape, 'output', out.shape, 'mu', mu.shape, 'var', var.shape)

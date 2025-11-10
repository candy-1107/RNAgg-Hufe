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
        # x: (B, C, H, W)
        b, c, h, w = x.size()
        y = x.view(b, c, -1).mean(dim=2)  # (B, C)
        y = self.fc1(y)
        y = F.relu(y)
        y = self.fc2(y)
        y = torch.sigmoid(y).view(b, c, 1, 1)
        return x * y


class ConvEncoder(nn.Module):
    def __init__(self, in_channels,  n_layers=3, input_hw=(64, 64)):
        super().__init__()
        # ensure each layer's out_channels = 2 * in_channels
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
        self.out_hw = (in_ch, H, W)  # in_ch is last "in_ch" after loop

    def forward(self, x):
        return self.net(x)


class ConvDecoder(nn.Module):
    def __init__(self, start_in_ch, n_layers=3):
        """Decoder that mirrors encoder doubling rule.

        start_in_ch should be the encoder's final channel count (encoder.out_hw[0]).
        The decoder will apply n_layers ConvTranspose2d layers, each halving the
        channel count (in_ch -> in_ch//2). After n_layers steps the output
        channels should match out_channels if start_in_ch == out_channels * 2**n_layers.
        """
        super().__init__()
        layers = []
        in_ch = start_in_ch
        for i in range(n_layers):
            out_ch = max(1, in_ch // 2)
            layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch
        # final out_ch should equal out_channels when start_in_ch is consistent
        # If not exact, a 1x1 conv could be used; here we keep the halving scheme.
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Conv_VAE(nn.Module):
    """Convolutional VAE with channel attention and 3-level encoder/decoder.

    This variant borrows the MLP-style linear stacks used in `MLP_VAE.py`:
    - After the conv encoder we flatten and pass through three linear+BN+ReLU layers
      before producing `mu` and `var` (linear outputs).
    - In the decoder we mirror this: a few linear layers expand the latent vector
      before a final linear that maps to the flattened conv shape which is then
      reshaped and passed through transpose-convs.

    Args:
      input_shape: (C, H, W)
      latent_dim: dimension of z
      base_channels: number of channels in first conv layer
      n_layers: number of down/up sampling layers (default 3 per requirement)
    """
    def __init__(self, input_shape, latent_dim=128, base_channels=32, n_layers=3, reduction=16, device='cpu'):
        super().__init__()
        self.device = device
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        C, H, W = input_shape
        self.n_layers = n_layers

        # attention on input channels
        self.attention = ChannelSELayer(C, reduction=reduction)

        # encoder convs
        self.encoder = ConvEncoder(C, n_layers=n_layers, input_hw=(H, W))
        enc_ch, enc_h, enc_w = self.encoder.out_hw
        # validate encoder spatial output is positive
        self.enc_shape = (enc_ch, enc_h, enc_w)
        enc_flat = enc_ch * enc_h * enc_w

        # MLP-style linear stack (encoder side) - sizes inspired by MLP_VAE
        # adapt if enc_flat is smaller than some intermediate size
        # keep final hidden dim 516 as in MLP_VAE
        hid1 = min(enc_flat, 4096)
        hid2 = min(max(516, hid1 // 2), 2048)
        hid3 = 516
        self.enc_fc1 = nn.Linear(enc_flat, hid1)
        self.enc_bn1 = nn.BatchNorm1d(hid1)
        self.enc_fc2 = nn.Linear(hid1, hid2)
        self.enc_bn2 = nn.BatchNorm1d(hid2)
        self.enc_fc3 = nn.Linear(hid2, hid3)
        self.enc_bn3 = nn.BatchNorm1d(hid3)

        # latent projections
        self.fc_mu = nn.Linear(hid3, latent_dim)
        self.fc_var = nn.Linear(hid3, latent_dim)

        # decoder MLP-style linear stack (mirror)
        self.dec_fc1 = nn.Linear(latent_dim, hid3)
        self.dec_bn1 = nn.BatchNorm1d(hid3)
        self.dec_fc2 = nn.Linear(hid3, hid2)
        self.dec_bn2 = nn.BatchNorm1d(hid2)
        self.dec_fc3 = nn.Linear(hid2, hid1)
        self.dec_bn3 = nn.BatchNorm1d(hid1)

        # final projection to flattened conv shape
        self.fc_decode = nn.Linear(hid1, enc_flat)
        # pass encoder's final channel count so decoder halves channels each layer
        self.decoder = ConvDecoder(C, start_in_ch=enc_ch, n_layers=n_layers)

    def reparameterize(self, mu, var):
        std = torch.sqrt(var + 1e-8)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        # x: (B, C, H, W)
        x = self.attention(x)
        h = self.encoder(x)
        b, c, h1, w1 = h.size()
        h_flat = h.view(b, -1)
        # MLP stack
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
        # decoder MLP
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
        return x_rec

    def forward(self, x):
        mu, var, conv_shape = self.encode(x)
        z = self.reparameterize(mu, var)
        x_rec = self.decode(z, conv_shape)
        return x_rec, mu, var



if __name__ == '__main__':
    # quick sanity check when run directly
    m = Conv_VAE((11, 64, 64), latent_dim=64, base_channels=8)
    x = torch.randn(2, 11, 64, 64)
    y, mu, var = m(x)
    print('sanity shapes ->', x.shape, '->', y.shape, mu.shape, var.shape, flush=True)

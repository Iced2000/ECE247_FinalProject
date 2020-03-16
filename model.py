import torch
import torch.nn as nn
import torch.nn.init as init

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)

class convModule(nn.Module):
    def __init__(self, inChannel, outChannel):
        super(convModule, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(inChannel, outChannel, kernel_size=3, padding=1),
            nn.BatchNorm2d(outChannel),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(outChannel, outChannel, kernel_size=3, padding=1),
            nn.BatchNorm2d(outChannel),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.module(x)

class Encoder(nn.Module):
    def __init__(self, inChannel, outChannel):
        super(Encoder, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder = convModule(inChannel, outChannel)

    def forward(self, x):
        enc = self.pool(x)
        return self.encoder(enc)

class Decoder(nn.Module):
    def __init__(self, inChannel, outChannel):
        super(Decoder, self).__init__()
        self.upSample = nn.ConvTranspose2d(inChannel, outChannel, kernel_size=2, stride=2)
        self.decoder = convModule(2 * outChannel, outChannel)

    def forward(self, x, resLink):
        dec = self.upSample(x)
        dec = torch.cat((dec, resLink), dim=1)
        return self.decoder(dec)

class UNet(nn.Module):
    def __init__(self, inChannel=1, outChannel=5, filters=32):
        super(UNet, self).__init__()
        self.encoder1 = convModule(inChannel, filters)
        self.encoder2 = Encoder(1 * filters, 2 * filters)
        self.encoder3 = Encoder(2 * filters, 4 * filters)
        self.encoder4 = Encoder(4 * filters, 8 * filters)
        self.encoder5 = Encoder(8 * filters, 16 * filters)

        self.decoder4 = Decoder(16 * filters, 8 * filters)
        self.decoder3 = Decoder(8 * filters, 4 * filters)
        self.decoder2 = Decoder(4 * filters, 2 * filters)
        self.decoder1 = Decoder(2 * filters, 1 * filters)
        self.outLayer = nn.Conv2d(1 * filters, outChannel, kernel_size=1)
        self.apply(weights_init)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        bottleNeck = self.encoder5(enc4)
        
        dec4 = self.decoder4(bottleNeck, enc4)
        dec3 = self.decoder3(dec4, enc3)
        dec2 = self.decoder2(dec3, enc2)
        dec1 = self.decoder1(dec2, enc1)
        out = self.outLayer(dec1)
        return out

class Discriminator(nn.Module):
    def __init__(self, channels=6):
        super(Discriminator, self).__init__()

        self.module = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=128, kernel_size=4, stride=4, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=4, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=4, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=(4, 2), stride=1, padding=0),
        )
        self.apply(weights_init)

    def forward(self, x):
        return self.module(x).view(-1, 1)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = UNet().to(device)
    #x = torch.randn(2, 1, 256, 128)
    #summary(model, (1, 256, 128))
    #dot = make_dot(model(x), params=dict(model.named_parameters()))
    #dot.render('tmp.gv', view=True)    
    #print(model(x).size())
    
    #model = Discriminator(6).to(device)
    #summary(model, (6, 256, 128))
    #x = torch.randn(2, 6, 256, 128)
    #print(model(x).size())
    
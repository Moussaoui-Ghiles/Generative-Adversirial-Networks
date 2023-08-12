import torch
import torch.nn as nn 


class CNNBlock(nn.Module):
    def __init__(self, in_chan, out_chan, stride = 2):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, 4, stride, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_chan),
            nn.LeakyReLU(0.2),
            
        )
    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, in_chan=3, features = [62, 128, 256, 512]):
        super().__init__()
        
        self.initial = nn.Sequential(
            nn.Conv2d(in_chan*2, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        
        layers = []
        
        in_chan = features[0]
        for feature in features[1:]:
            layers.append(
                CNNBlock(in_chan, feature, stride = 1 if feature == features[-1] else 2),
                
            )   
            in_chan = feature
        layers.append(
            nn.Conv2d(in_chan, 1, kernel_size=4, stride = 1, padding_mode="reflect")
            
        ) 
        self.model = nn.Sequential(*layers)
    def forward(self, x, y):
        x = torch.cat([x,y], dim =1)
        x = self.initial(x)
        return self.model(x)
    
def test():
    x = torch.randn(1, 3, 256, 256)
    y = torch.randn(1, 3, 256, 256)
    model = Discriminator()
    preds = model(x,y)
    print(preds.shape)

       
test()
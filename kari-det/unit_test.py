import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        sub_module0 = nn.ModuleList([nn.Conv2d(3,2,1), nn.Conv2d(2,2,1), nn.Conv2d(2,2,1)])
        module0 = nn.ModuleList([sub_module0, nn.Conv2d(2,2,1), nn.Conv2d(2,2,1)])
        module0.append(nn.Conv2d(2,2,1))
        self.model = nn.Sequential(*module0)
        #self.model.add_module("2", nn.Conv2d(2,2,1))
        self.conv2 = nn.Conv2d(2,1,1)
    def forward(self, x):
        y = self.model(x)
        y = self.conv2(y)
        return y


model = MyModel()
print([k for k, v in model.named_parameters()])




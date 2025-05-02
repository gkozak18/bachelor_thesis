import torch
import torch.nn as nn
import torchvision

def load_model(path="densenet121_testloss8.7.pth"):
    model = torchvision.models.densenet121()
    model.classifier = torch.nn.Linear(1024, 1)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

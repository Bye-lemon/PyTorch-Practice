import torch
import torchvision

from models.CNN import AlexNet

my_alex = AlexNet()
alex = torchvision.models.alexnet(pretrained=True)
my_alex.fc3.weight = alex.classifier[6].weight
print(my_alex.fc3.weight)

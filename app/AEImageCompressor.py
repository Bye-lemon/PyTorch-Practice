import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchnet as tnt
import os.path as op
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from models.AE import *

LOG = print
WATCH = lambda x: print(x.shape)

# Hyper Parameters
BASE_EPOCH = 10
EPOCH = 10
BATCH_SIZE = 16
LEARNING_RATE = 0.0045
DOWNLOAD_CIFAR10 = False
LOG_STEP = 1000
MODE = "test"
DATA_PATH = "../data/"
MODEL_PATH = "../logs/AE/Epoch_13_AvgLoss_0.009554789398890044.pth"
LOG_PATH = "../logs/AE/"

# Device Settings
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
LOG(f"[DEVICE] Device {device} is ready.")

# Load Data
transform = transforms.Compose([
    transforms.ToTensor(),
])
trainset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True, download=DOWNLOAD_CIFAR10, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=False, download=DOWNLOAD_CIFAR10, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
LOG(f"[DATA]   Finished loading data.")

# Model Definition
model = AutoEncoder().to(device)
if MODEL_PATH is not None:
    model.load_state_dict(torch.load(MODEL_PATH))
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()
loss_meter = tnt.meter.AverageValueMeter()
LOG(f"[MODEL]  Model building complete.")

# Train
if MODE == "train":
    for epoch in range(EPOCH):
        loss_meter.reset()
        for index, data in tqdm.tqdm(enumerate(trainloader)):
            input_, _ = data
            input_ = torch.autograd.Variable(input_).to(device)

            optimizer.zero_grad()

            decode = model(input_)
            loss = criterion(decode.view(BATCH_SIZE, -1), input_.view(BATCH_SIZE, -1))
            loss.backward()
            optimizer.step()
            loss_meter.add(loss.item())

            if index % LOG_STEP == 0:
                print(f"Epoch {BASE_EPOCH + epoch + 1} Batch {index + 1} Average loss {loss_meter.value()[0]}")

        torch.save(model.state_dict(),
                   op.join(LOG_PATH, f"Epoch_{BASE_EPOCH + epoch + 1}_AvgLoss_{loss_meter.value()[0]}.pth"))

# Test
if MODE == "test":
    imshow = lambda img: plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))

    dataiter = iter(testloader)
    images, labels = dataiter.next()

    with torch.no_grad():
        input_ = images.to(device)
        output = model(input_)

    imshow(torchvision.utils.make_grid(images.cpu()))
    plt.show()
    imshow(torchvision.utils.make_grid(output.cpu()))
    plt.show()

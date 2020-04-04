import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchnet as tnt
import os
import os.path as op
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from models.GAN import *

LOG = print
WATCH = lambda x: print(x.shape)

# Hyper Parameters
BASE_EPOCH = 0
EPOCH = 10
BATCH_SIZE = 16
NOISE_DIMENSION = 100
NUM_FEATURES = 64 * 64
LEARNING_RATE = 0.0004
DOWNLOAD_CIFAR10 = False
LOG_STEP = 400
MODE = "train"
DATA_PATH = "../data/"
DIS_MODEL_PATH = None
GEN_MODEL_PATH = None
LOG_PATH = "../logs/GAN/"

# Device Settings
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
LOG(f"[DEVICE] Device {device} is ready.")

# Log Directory Checking
if not op.exists(LOG_PATH):
    os.mkdir(LOG_PATH)

# Load Data
transform = transforms.Compose([
    transforms.ToTensor(),
])
trainset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True, download=DOWNLOAD_CIFAR10, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
LOG(f"[DATA]   Finished loading data.")

# Model Definition
dis = Discriminator().to(device)
if DIS_MODEL_PATH is not None:
    dis.load_state_dict(torch.load(DIS_MODEL_PATH))
dis_optimizer = optim.Adam(dis.parameters(), lr=LEARNING_RATE)
dis_meter = tnt.meter.AverageValueMeter()
dis_real_meter = tnt.meter.AverageValueMeter()
dis_fake_meter = tnt.meter.AverageValueMeter()

gen = Generator(in_features=NOISE_DIMENSION, num_features=NUM_FEATURES).to(device)
if GEN_MODEL_PATH is not None:
    gen.load_state_dict(torch.load(GEN_MODEL_PATH))
gen_optimizer = optim.Adam(gen.parameters(), lr=LEARNING_RATE)
gen_meter = tnt.meter.AverageValueMeter()

criterion = nn.BCELoss()
LOG(f"[MODEL]  Model building complete.")

# Train
if MODE == "train":
    for epoch in range(EPOCH):
        dis_meter.reset()
        dis_real_meter.reset()
        dis_fake_meter.reset()
        gen_meter.reset()
        for index, data in tqdm.tqdm(enumerate(trainloader)):
            input_, _ = data
            input_ = torch.autograd.Variable(input_).to(device)

            # Train Discriminator
            # Generate real label, fake label and noise vector
            real_labels = torch.autograd.Variable(torch.ones(input_.shape[0], 1)).to(device)
            fake_labels = torch.autograd.Variable(torch.zeros(input_.shape[0], 1)).to(device)
            noise = torch.autograd.Variable(torch.randn(input_.shape[0], NOISE_DIMENSION)).to(device)
            # Use real image to train the network
            real_out = dis(input_)
            dis_real_loss = criterion(real_out, real_labels)
            dis_real_meter.add(dis_real_loss.item())
            # Use fake image to train the network
            fake_out = dis(gen(noise).detach())
            dis_fake_loss = criterion(fake_out, fake_labels)
            dis_fake_meter.add(dis_fake_loss.item())
            # Backprop and optimize
            dis_optimizer.zero_grad()
            dis_loss = dis_real_loss + dis_fake_loss
            dis_loss.backward()
            dis_optimizer.step()
            dis_meter.add(dis_loss.item())

            # Train Generator
            # Generate noise vector
            noise = torch.autograd.Variable(torch.randn(input_.shape[0], NOISE_DIMENSION)).to(device)
            # Use noise vector to train the network
            gen_out = dis(gen(noise))
            gen_loss = criterion(gen_out, real_labels)
            # Backprop and optimize
            gen_optimizer.zero_grad()
            gen_loss.backward()
            gen_optimizer.step()
            gen_meter.add(gen_loss.item())

            # Logging
            if index % LOG_STEP == 0:
                print(
                    f"Epoch {BASE_EPOCH + epoch + 1} Batch {index + 1} Discriminator loss {dis_meter.value()[0]} Generator loss {gen_meter.value()[0]} Real loss {dis_real_meter.value()[0]} Fake loss {dis_fake_meter.value()[0]}")

        torch.save(dis.state_dict(),
                   op.join(LOG_PATH, f"DIS_Epoch_{BASE_EPOCH + epoch + 1}_AvgLoss_{dis_meter.value()[0]}.pth"))
        torch.save(gen.state_dict(),
                   op.join(LOG_PATH, f"GEN_Epoch_{BASE_EPOCH + epoch + 1}_AvgLoss_{gen_meter.value()[0]}.pth"))

# Generate
if MODE == "generate":
    imshow = lambda img: plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
    noise = torch.autograd.Variable(torch.randn(1, NOISE_DIMENSION)).to(device)

    with torch.no_grad():
        output = gen(noise)

    imshow(torchvision.utils.make_grid(output.cpu()))
    plt.show()

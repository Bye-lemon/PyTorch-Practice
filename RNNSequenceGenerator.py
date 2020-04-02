import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchnet as tnt
import tqdm
import os.path as op
import numpy as np

from models.RNN import *

LOG = print
WATCH = lambda x: print(x.shape)

# Hyper Parameters
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCH = 10
BATCH_SIZE = 32
MAX_SEQUENCE_LENGTH = 125
MAX_GENERIC_SEQUENCE_LENGTH = 64
LOG_STEP = 40
MODE = "interface"
DATA_PATH = "data/tang.npz"
MODEL_PATH = None
LOG_PATH = "logs/"

# Device Settings
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
LOG(f"[DEVICE] Device {device} is ready.")

# Load Data
raw_data = np.load(DATA_PATH, allow_pickle=True)
data, word2idx, idx2word = raw_data["data"], raw_data["word2ix"].item(), raw_data["ix2word"].item()
dataloader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
LOG(f"[DATA]   Data is loaded. Vocabulary size is {len(word2idx)}")

# Model Definition
model = RNN(vocab_size=len(word2idx), embedding_dim=128, hidden_dim=256, num_layers=2, target="lstm")
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss()
loss_meter = tnt.meter.AverageValueMeter()
if MODEL_PATH is not None:
    model.load_state_dict(torch.load(MODEL_PATH))
model.to(device)
LOG(f"[MODEL]  Build model complete.")

# Train
if MODE == "train":
    for epoch in range(EPOCH):
        loss_meter.reset()
        for index, data in tqdm.tqdm(enumerate(dataloader, 0)):
            data = data.long().contiguous().to(device)

            optimizer.zero_grad()

            input_, target = data[:, : -1], data[:, 1:]
            output, _ = model(input_)
            loss = criterion(output, target.reshape(-1))
            loss.backward()
            optimizer.step()
            loss_meter.add(loss.item())

            if (index + 1) % LOG_STEP == 0:
                print(f"Epoch {epoch + 1} Batch {index + 1} Average loss {loss_meter.value()[0]}")

        torch.save(model.state_dict(), op.join(LOG_PATH, f"Epoch_{epoch}_AvgLoss_{loss_meter.value()[0]}.pth"))


# Interface Settings
START_WORD = "漂泊不见长安月"
USE_PREFIX = True
PREFIX_WORD = "落霞与孤鹜齐飞，秋水共长天一色。"


# Interface
if MODE == "interface":
    gen_word = list(START_WORD)
    start_word_length = len(gen_word)
    input = torch.Tensor([word2idx["<START>"]]).view(1, 1).long().to(device)
    hidden = None
    if USE_PREFIX:
        for word in PREFIX_WORD:
            output, hidden = model(input, hidden)
            input = input.data.new([word2idx[word]]).view(1, 1)

    for i in range(MAX_GENERIC_SEQUENCE_LENGTH):
        output, hidden = model(input, hidden)
        if i < start_word_length:
            input = input.data.new([word2idx[gen_word[i]]]).view(1, 1)
        else:
            top_index = output.data[0].topk(1)[1][0].item()
            if idx2word[top_index] != "<EOP>":
                gen_word.append(idx2word[top_index])
                input = input.data.new([top_index]).view(1, 1)
            else:
                break

    print("".join(gen_word))


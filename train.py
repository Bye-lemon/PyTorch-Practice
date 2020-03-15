import torch.optim as optim
import torchvision.transforms as transforms
import torch.utils as utils
import time
import tqdm

from models.CNN import *
from dataset import MyDataSet

# Hyper Parameters
EPOCH = 8
BATCH_SIZE = 4
LEARNING_RATE = 0.0005
SPLIT_RATE = 0.08
MODE = "train"

transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.Resize((227, 227)),
    transforms.ToTensor()
])
datasets = MyDataSet(root=".\\data\\RM", datatxt="data.txt", transforms=transforms)
train_size = int(SPLIT_RATE * len(datasets)) - int(SPLIT_RATE * len(datasets)) % BATCH_SIZE
test_size = len(datasets) - train_size
train_set, test_set = torch.utils.data.random_split(datasets, [train_size, test_size])
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=True)

net = AlexNet()
net.fc3 = nn.Linear(in_features=4096, out_features=6)
net.cuda()

print(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9)

for epoch in range(EPOCH):
    running_loss = .0
    for index, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = torch.autograd.Variable(inputs).cuda(), torch.autograd.Variable(labels).cuda()

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if index % 100 == 99:
            print(f"Epoch {epoch + 1} Batch {index + 1} Average loss {running_loss / 100}")
            running_loss = .0

torch.save(net.state_dict(), "./alexnet.pth")

"""
net.load_state_dict(torch.load("./alexnet.pth"))

correct = 0
total = 0
start = time.time()
with torch.no_grad():
    for data in tqdm.tqdm(test_loader):
        images, labels = data
        images, labels = torch.autograd.Variable(images).cuda(), torch.autograd.Variable(labels).cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the {test_size} test images: %d %%' % (
        100 * correct / total))
print(f"Test per image need %f" % ((time.time() - start) / float(test_size)))
"""

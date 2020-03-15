import torch
import os.path as op
import PIL.Image
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    def __init__(self, root, datatxt, transforms=None):
        super(MyDataSet, self).__init__()
        self.imgs = []
        with open(op.join(root, datatxt)) as f:
            for line in f:
                data = line.rstrip().split()
                self.imgs.append((data[0], data[1]))
        self.root = root
        self.transform = transforms

    def __getitem__(self, index):
        filename, label = self.imgs[index]
        img = PIL.Image.open(op.join(self.root, filename)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    import torchvision.transforms as transforms
    import torch.utils as utils

    transforms = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor()
    ])
    datasets = MyDataSet(root="./data/RM", datatxt="data.txt", transforms=transforms)
    dataloader = utils.data.DataLoader(datasets, batch_size=4, shuffle=True)
    for data, label in dataloader:
        print(data)
        print(label)
        break
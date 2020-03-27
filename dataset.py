import torch
import tqdm
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
        self.means = None
        self.stds = None

    def __getitem__(self, index):
        filename, label = self.imgs[index]
        img = PIL.Image.open(op.join(self.root, filename)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

    def get_mean_std(self):
        if self.means is not None and self.stds is not None:
            return self.means, self.stds
        else:
            self.means, self.stds = torch.tensor([.0, .0, .0])\
                , torch.tensor([.0, .0, .0])
            for filename, label in tqdm.tqdm(self.imgs):
                img = PIL.Image.open(op.join(self.root, filename)).convert("RGB")
                if self.transform is not None:
                    img = self.transform(img)
                for channel in range(3):
                    self.means[channel] += img[channel, :, :].mean()
                    self.stds[channel] += img[channel, :, :].std()
            self.means /= len(self.imgs)
            self.stds /= len(self.imgs)
            return self.means, self.stds


if __name__ == "__main__":
    import torchvision.transforms as transforms
    import torch.utils as utils

    transforms = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor()
    ])
    datasets = MyDataSet(root="./data/RM", datatxt="data.txt", transforms=transforms)
    dataloader = utils.data.DataLoader(datasets, batch_size=4, shuffle=True)
    print(datasets.get_mean_std())
    for data, label in dataloader:
        print(data)
        print(label)
        break

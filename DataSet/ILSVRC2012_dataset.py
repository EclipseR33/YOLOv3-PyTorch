import torch
from torch.utils.data import Dataset
from PIL import Image

import os
from glob import glob
from tqdm import tqdm


class ILSVRCDataSet(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.img_root = os.path.join(root, 'ILSVRC2012_img_train')
        self.transform = transform

        classes = glob(os.path.join(self.img_root, '*'))[:10]
        img_path = []
        label = []
        pbar = tqdm(classes)
        for i, path in enumerate(pbar):
            names = glob(os.path.join(path, '*'))
            l = path.split('\\')[-1]
            l = [l for _ in range(len(names))]

            img_path = img_path + names
            label = label + l
        self.img_set = {'image': img_path, 'label': label}
        self.classes = [c.split('\\')[-1] for c in classes]

    def __len__(self):
        return len(self.img_set['image'])

    def __getitem__(self, idx):
        image = Image.open(self.img_set['image'][idx])
        label = self.classes.index(self.img_set['label'][idx])

        image = self.transform(image)
        if image.size()[0] == 1:
            image = torch.zeros((3, image.size(1), image.size(2))) + image
        image /= 255.0

        label = torch.tensor(label)
        return image, label


if __name__ == '__main__':
    from torchvision import transforms
    dataset_path = 'ILSVRCDataSet.pt'
    root = r'D:\AI\Dataset\ILSVRC2012'
    dataset = ILSVRCDataSet(root, transform=transforms.ToTensor())
    torch.save(dataset, dataset_path)
    x = dataset[0]

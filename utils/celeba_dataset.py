from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random


class CelebADataset(Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, selected_attr, protected_attr, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attr = selected_attr
        self.protected_attr = protected_attr
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[0].split(',')
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split(',')
            filename = split[0]
            values = split[0:]
            idx = self.attr2idx[self.selected_attr]
            protected_idx = self.attr2idx[self.protected_attr]
            
            if values[idx] == '1':
                label = int(1)
            else:
                label = int(0)
               
            if values[protected_idx] == '1':
                protected_label = int(1)
            else:
                protected_label = int(0)

            # Use an 80/20 train/test split. With 30,000 in the dataset this is 6,000 in test. 
            if (i+1) < 6000:
                self.test_dataset.append([filename, protected_label, label])
            else:
                self.train_dataset.append([filename, protected_label, label])
            

        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, protected_label, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), protected_label, label

    def __len__(self):
        """Return the number of images."""
        return self.num_images
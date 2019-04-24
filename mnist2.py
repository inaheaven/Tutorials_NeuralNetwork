import torch
import torch.nn.init
from torch.autograd import Variable

import torchvision.utils as utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import random

# Loading MNIST dataset
mnist_train = dsets.MNIST(root='data/', train=True, transform=transforms.ToTensor(), download=True)
mnist_test = dsets.MNIST(root='data/', train=False, transform=transforms.ToTensor(), download=True)
print(mnist_train.train_data.size())
print(mnist_test.test_data.size())

idx=0
plt.imshow(mnist_train.train_data[idx,:,:].numpy(), cmap='gray')
plt.title('%i'% mnist_train.train_labels[idx])
plt.show()

batch_size = 100
data_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True, num_workers=1)

def imshow(img):
    imig = img/2 +0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

batch_images, batch_labels = next(iter(data_loader))
print(batch_images.size())
print(batch_labels.size())

linear1 = torch.nn.Linear(784, 512, bias = True)
linear2 = torch.nn.Linear(512, 10, bias = True)
relu = torch.nn.ReLU()

model = torch.nn.Sequential(linear1, relu, linear2)

print(model
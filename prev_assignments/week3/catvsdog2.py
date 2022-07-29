'''
couldn't train this model 
tried various combinations of neural nets, increasing decreasing features
accuracy comes to be 50%
finally i tried using tanh instead of relu
getting around 61% accuracy
still not achieving that 90+ % capability of cnn
had put so many pth files, most of them are plain useless with 50% accuracy 
'''

import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from torch.utils.data import Dataset

ROWS = 64
COLS = 64
CHANNELS = 3

num_epochs = 1000
learning_rate = 0.0001
batch_size = 1000
datapath = "../PetImages/"

cat_data = [datapath + "Cat/" + i for i in os.listdir(datapath + "Cat/")]
dog_data = [datapath + "Dog/" + i for i in os.listdir(datapath + "Dog/")]

cats_train = cat_data[:12000]
dogs_train = dog_data[:12000]
cats_test = cat_data[12000:]
dogs_test = dog_data[12000:]


def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)


class catvdogData(Dataset):

    def __init__(self, cats, dogs):

        m_cats = len(cats)
        m_dogs = len(dogs)
        m = m_cats + m_dogs
        X_train = np.zeros((m, ROWS, COLS, CHANNELS), dtype=np.uint8)
        y = np.zeros((m))

        for i, image_file in enumerate(cats):
            X_train[i, :] = read_image(image_file)
            y[i] = 0

        for i, image_file in enumerate(dogs):
            X_train[i + m_cats, :] = read_image(image_file)
            y[i + m_cats] = 1
        
        X_train = X_train.transpose((0, 3, 1, 2))

        X_train = torch.from_numpy(X_train.astype(np.float32))
        y_train = torch.from_numpy(y.astype(np.int64))

        self.n_samples = m
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


train_loader = torch.utils.data.DataLoader(dataset=catvdogData(cats_train, dogs_train),
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=catvdogData(cats_test, dogs_test),
                                          batch_size=batch_size,
                                          shuffle=True)

# 1) Model
class Convnet(nn.Module):
    def __init__(self):
        super(Convnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 9, 7)
        self.pool = nn.MaxPool2d(2,2)

        self.tanh = nn.Tanh()

        self.fc1 = nn.Linear(6*4*4, 2)

    def forward(self, x):
        # 3, 64, 64
        out = self.pool(self.tanh(self.conv1(x)))  # 6, 4, 4

        out = out.view(-1, 6*4*4)

        out = self.fc1(out)

        return out

PATH = './catvsdog2.pth'

model = Convnet()
model.load_state_dict(torch.load(PATH))

# 2) Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
# 3) Training loop
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 12 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

    if (epoch+1) % 50 == 0:
        torch.save(model.state_dict(), PATH)
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            for images, labels in test_loader:
                outputs = model(images)
                # max returns (value ,index)
                _, predicted = torch.max(outputs.data, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

            acc = 100.0 * n_correct / n_samples
            print(f'Accuracy of the network on the {n_samples} test images: {acc} %')

print('Finished Training')

torch.save(model.state_dict(), PATH)

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in train_loader:
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        # print(images)
        # print(predicted, labels)

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the {n_samples} test images: {acc} %')

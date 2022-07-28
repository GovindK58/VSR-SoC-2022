'''
finally did single layer logistic regression(3*64*64 -> 2)
accuracy  achieved = 68.92%
earlier couldn't 
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

num_epochs = 20
learning_rate = 0.001
batch_size = 100
datapath = "../PetImages/"

cats_train = [datapath + "Cat/" +
              i for i in os.listdir(datapath + "Cat/")][:12000]
dogs_train = [datapath + "Dog/" +
              i for i in os.listdir(datapath + "Dog/")][:12000]
cats_test = [datapath + "Cat/" +
             i for i in os.listdir(datapath + "Cat/")][12000:]
dogs_test = [datapath + "Dog/" +
             i for i in os.listdir(datapath + "Dog/")][12000:]


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

        # X_train = X_train.view(m, CHANNELS, ROWS, COLS)\
        
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
class Mymodel(nn.Module):
    def __init__(self):
        super(Mymodel, self).__init__()

        self.relu = nn.ReLU()
        self.fc = nn.Linear(3*64*64, 2)

    def forward(self, x):
        # 3, 64, 64
        out = x.view(-1, 3*64*64)
        out = self.fc(out)
        return out

PATH = './catvsdog.pth'

model =  Mymodel()
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

        if (i+1) % 100 == 0:
            print(
                f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
                

print('Finished Training')

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
        # print(images)
        # print(predicted, labels)

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the {n_samples} test images: {acc} %')

'''
1 connet + 2 fully connected layers (1*28*28 -> 6*12*12 -> 200 -> 10)
recieved accuracy = 98.6%
loss was 0.00005 in last epoch but still couldn't get more accuracy
'''

from re import X
from keras.datasets import mnist
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset

(X_train, y_train), (X_test, y_test) = mnist.load_data()

input_size = 784
num_classes = 10
learning_rate = 0.00001
num_epochs = 2
batch_size = 1000

class mnistDataset(Dataset):

    def __init__(self, x, y):

        X_train = torch.from_numpy(x.astype(np.float32))  
        y_train = torch.from_numpy(y.astype(np.int64))
        X_train = X_train.view(y.size, 1, 28, 28)

        self.n_samples = y.size
        self.x_data = X_train 
        self.y_data = y_train 

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

train_loader = torch.utils.data.DataLoader(dataset=mnistDataset(X_train, y_train), 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=mnistDataset(X_test, y_test), 
                                          batch_size=batch_size, 
                                          shuffle=False)

# 1) Model

class Convnet(nn.Module):
    def __init__(self):
        super(Convnet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2,2)

        self.fc1 = nn.Linear(12*12*6, 200)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(200, 10)
    
    def forward(self, x):
        # 1, 28, 28
        out = self.relu(self.conv1(x))  # 6, 24, 24
        out = self.pool(out) # 6, 12, 12
        out = out.view(-1, 12*12*6)

        out = self.relu(self.fc1(out))
        out = self.fc2(out)

        return out

PATH = './mnistcnn.pth'
model = Convnet()
model.load_state_dict(torch.load(PATH))

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

# Train the model
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 20 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.6f}')

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

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the {n_samples} test images: {acc} %')
'''
2 layer neural net for mnist data (784 -> 300 -> 10)
recieved accuracy = 94.56%
'''

from keras.datasets import mnist
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset

(X_train, y_train), (X_test, y_test) = mnist.load_data()

input_size = 784
num_classes = 10
learning_rate = 0.003
num_epochs = 5
batch_size = 10000

class mnistDataset(Dataset):

    def __init__(self, x, y):

        X_train = np.reshape(x, (-1, 784))
        X_train = torch.from_numpy(X_train.astype(np.float32))  
        y_train = torch.from_numpy(y.astype(np.int64)) # was using float32 earlier, and spent lot of time to find this error as Crossentropyloss requires integer

        self.n_samples = y.size
        self.x_data = X_train # size [n_samples, n_features]
        self.y_data = y_train # size [n_samples, 1]

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

class Neuralnet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Neuralnet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, 300)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(300, num_classes)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

PATH = './mnist.pth'
model = Neuralnet(input_size, num_classes)
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

    print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

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

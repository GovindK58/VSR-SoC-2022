'''
code seems to be working and i saved model in this directory but i haven't tested the model yet.
getting the data in the required form is the hardest task
loss comes out to be around 50
but this is just a logistic regression model
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

n_features = 64*64*3
batch_size = 100
datapath = "./PetImages/"

cats = [datapath + "Cat/" + i for i in os.listdir(datapath + "Cat/")]
dogs = [datapath + "Dog/" + i for i in os.listdir(datapath + "Dog/")]

def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)

class catvdogData(Dataset):

    def __init__(self):

        m_cats = len(cats)
        m_dogs = len(dogs)
        m = m_cats + m_dogs
        X = np.zeros((m, ROWS, COLS, CHANNELS),dtype=np.uint8)
        y = np.zeros((m))

        for i, image_file in enumerate(cats):
            X[i,:] = read_image(image_file)
            y[i] = 0

        for i, image_file in enumerate(dogs):
            X[i + m_cats,:] = read_image(image_file)
            y[i + m_cats] = 1
        
        # print(X.size)
        X_train = np.reshape(X, (y.size, CHANNELS*COLS*ROWS))
        # print(X_train.size)
        # print(y.size)

        X_train = torch.from_numpy(X_train.astype(np.float32))  
        y_train = torch.from_numpy(y.astype(np.float32))
        y_train = y_train.view(y_train.shape[0], 1)
        self.n_samples = y.size
        self.x_data = X_train # size [n_samples, n_features]
        self.y_data = y_train # size [n_samples, 1]


    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

train_loader = torch.utils.data.DataLoader(dataset=catvdogData(), 
                                           batch_size=batch_size, 
                                           shuffle=True)

# 1) Model
# Linear model f = wx + b , sigmoid at the end
class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

PATH = './catvdog.pth'

model = Model(n_features)
model.load_state_dict(torch.load(PATH))

# 2) Loss and optimizer
num_epochs = 2
learning_rate = 0.003
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


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
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

print('Finished Training')

torch.save(model.state_dict(), PATH)


# with torch.no_grad():
#     n_correct = 0
#     n_samples = 0
#     for images, labels in test_loader:
#         predicted = model(images)
#         print(predicted)
#         print(labels)
#         n_samples += labels.size(0)
#         n_correct += (predicted == labels).sum().item()

#     acc = 100.0 * n_correct / n_samples
#     print(f'Accuracy of the network on the {n_samples} test images: {acc} %')

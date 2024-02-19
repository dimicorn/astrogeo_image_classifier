import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from model import CNN
from tqdm import tqdm


num_epochs = 11
num_classes = 5
batch_size = 64
learning_rate = 0.002

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# TODO: Add other transformations
transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
data = datasets.ImageFolder('aug', transform=transform)

train_size = int(0.8 * len(data))
test_size = len(data) - train_size
train, test = torch.utils.data.random_split(data, [train_size, test_size])
trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)

model = CNN()
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

model.train()
for epoch in range(num_epochs):
    train_loss = 0.0
    for data, target in tqdm(trainloader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
    train_loss = train_loss/len(trainloader.dataset)
    print(f'Epoch: {epoch+1} \tTraining Loss: {train_loss:.6f}')

y_test, predictions = [], []
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in tqdm(testloader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        y_test.extend(labels.cpu().numpy())
        predictions.extend(predicted.cpu().numpy())
    print(f'Accuracy of the network on the {total} test images: {100 * correct / total:.3f} %')

torch.save(model.state_dict(), f'./models/test.pth')
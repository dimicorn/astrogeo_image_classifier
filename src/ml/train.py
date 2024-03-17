import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import matplotlib.pyplot as plt
from ml.model import CNN


def train_model(
        model_name: str, model_tensor_name: str, num_epochs: int = 11,
        batch_size: int = 64, learning_rate: float = 0.002,
        train_test_ratio: float = 0.8, data_path: str = 'aug'
    ) -> None:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # TODO: Add other transformations
    transform = transforms.Compose(
        [transforms.Grayscale(),
         transforms.RandomAffine(degrees=(0, 180), translate=(0.1, 0.3)), #
         transforms.ToTensor()]
    )
    data = datasets.ImageFolder(data_path, transform=transform)

    train_size = int(train_test_ratio * len(data))
    test_size = len(data) - train_size
    train, test = torch.utils.data.random_split(data, [train_size, test_size])
    trainloader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=True
    )

    if model_name == 'cnn':
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
        acc = 100 * correct / total
        print(
            f'Accuracy of the network on the {total} test images:'
            f'{acc:.3f} %'
        )

    torch.save(
        model.state_dict(),
        f'src/cnn/models/{model_tensor_name}_{acc:.3f}.pth'
    )

    cm = confusion_matrix(y_test, predictions)
    ConfusionMatrixDisplay(cm).plot()
    plt.savefig(f'src/cnn/models/ConfMatrix_{model_tensor_name}_{acc:.3f}.png', dpi=500)

    precision = precision_score(y_test, predictions, average='macro')
    recall = recall_score(y_test, predictions, average='macro')
    
    f1 = 2 * precision * recall / (precision + recall)
    f1_macro = f1_score(y_test, predictions, average='macro')
    f1_weighted = f1_score(y_test, predictions, average='weighted')
    print(f'Precision: {100 * precision:.3f} %')
    print(f'Recall: {100 * recall:.3f} %')
    print(f'F1 Score: {100 * f1:.3f} %')
    print(f'F1 Macro Score: {100 * f1_macro:.3f} %')
    print(f'F1 Weighted Score: {100 * f1_weighted:.3f} %')
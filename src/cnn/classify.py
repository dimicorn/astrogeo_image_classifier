import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
import json
from tqdm import tqdm
from model import CNN


class AstrogeoDataset(Dataset):
    def __init__(self, dir: str, transform=None) -> None:
        self.transform = transform
        self.images = os.listdir(dir)
        self.dir = dir

    def __getitem__(self, index: int) -> tuple:
        image = read_image(f'{self.dir}/{self.images[index]}', mode=ImageReadMode.RGB)
        file_name = self.images[index]
        if self.transform is not None:
            image = self.transform(image)
        return (file_name, image)

    def __len__(self) -> int:
        return len(self.images)


batch_size = 32
transform = transforms.Compose(
    [transforms.ToPILImage(), transforms.Resize((512, 512)), transforms.Grayscale(), transforms.ToTensor()]
)
val = AstrogeoDataset('data', transform=transform)
valloader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CNN()
model.to(device)
model.load_state_dict(torch.load('src/cnn/models/colab0.pth'))
model.eval()

val_preds = {}
with torch.no_grad():
    for file_names, images in tqdm(valloader):
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        val_preds.update(dict(zip(file_names, predicted.cpu().tolist())))

    with open('src/cnn/val_preds.json', 'w') as f:
        json.dump(val_preds, f)
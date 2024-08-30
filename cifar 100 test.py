import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing


# Define classes
classes = ('beaver', 'dolphin', 'otter', 'seal', 'whale', 'aquarium fish', 'flatfish', 'ray', 'shark', 'trout',
           'orchids', 'poppies', 'roses', 'sunflowers', 'tulips', 'bottles', 'bowls', 'cans', 'cups', 'plates',
           'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers', 'clock', 'computer keyboard', 'lamp',
           'telephone', 'television', 'bed', 'chair', 'couch', 'table', 'wardrobe', 'bee', 'beetle', 'butterfly',
           'caterpillar', 'cockroach', 'bear', 'leopard', 'lion', 'tiger', 'wolf', 'bridge', 'castle', 'house',
           'road', 'skyscraper', 'cloud', 'forest', 'mountain', 'plain', 'sea', 'camel', 'cattle', 'chimpanzee',
           'elephant', 'kangaroo', 'fox', 'porcupine', 'possum', 'raccoon', 'skunk', 'crab', 'lobster', 'snail',
           'spider', 'worm', 'baby', 'boy', 'girl', 'man', 'woman', 'crocodile', 'dinosaur', 'lizard', 'snake',
           'turtle', 'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel', 'maple', 'oak', 'palm', 'pine', 'willow',
           'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train', 'lawn-mower', 'rocket', 'streetcar', 'tank',
           'tractor')

# Define CNN architecture
class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1)
            self.conv3 = torch.nn.Conv2d(64, 128, 3, padding=1)
            self.pool = torch.nn.MaxPool2d(2, 2)
            self.fc1 = torch.nn.Linear(128 * 4 * 4, 512)
            self.fc2 = torch.nn.Linear(512, 100)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = x.view(-1, 128 * 4 * 4)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x


def main():
    # Load the saved model
    saveLocation = './cifar_net_100classes.pth'
    net = Net()
    net.load_state_dict(torch.load(saveLocation))
    net.eval()  # Set the model to evaluation mode

    # Define transform for normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Function to show images
    def imshow(img):
        img = img / 2 + 0.5  # Unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # Load testset
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    # Load test images
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    # Define a function to show images
    def imshow(img):
        img = img / 2 + 0.5  # Unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # Show test images
    imshow(torchvision.utils.make_grid(images))

    # Predict labels for test images
    with torch.no_grad():
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)

    # Display predicted labels
    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))

if __name__ == '__main__':
    main()

import torch
import torchvision
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from torchvision import datasets
import numpy as np

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from config import *

import albumentations as A
from albumentations.pytorch import ToTensorV2


class AlbumentationsCIFAR10Wrapper(Dataset):
    def __init__(self, root='./data', train=True, download=True):
        self.data = datasets.CIFAR10(root=root, train=train, download=download)
        self.train = train
        self.transform = A.Compose([
            A.Normalize(mean, std),
            A.PadIfNeeded(min_height=40, min_width=40, always_apply=True),  # 2 x 4 = 8 on each side
            A.RandomCrop(height=32, width=32, always_apply=True),
            A.HorizontalFlip(p=0.5),
            A.CoarseDropout(max_holes=1, max_height=8, max_width=8, min_holes=1, min_height=8, min_width=8, fill_value=mean, mask_fill_value=None),
            ToTensorV2()
        ])

        self.test_transforms = A.Compose([
            A.Normalize(mean, std),
            ToTensorV2()
        ])

    def __getitem__(self, idx):
        img, label = self.data[idx]
        img = np.array(img)  # PIL Image to numpy array

        transform_fn = self.transform
        if self.train == False:
            transform_fn = self.test_transforms
        augmented = transform_fn(image=img)
        img = augmented['image']
        return img, label

    def __len__(self):
        return len(self.data)
    

def get_train_and_test_data(mean = [0.4914, 0.4822, 0.4465], std = [0.247, 0.243, 0.261]):
    train_transforms = A.Compose([
        A.Normalize(mean, std),
        A.PadIfNeeded(min_height=40, min_width=40, always_apply=True),  # 2 x 4 = 8 on each side
        A.RandomCrop(height=32, width=32, always_apply=True),
        A.CoarseDropout(max_holes=1, max_height=8, max_width=8, min_holes=1, min_height=8, min_width=8, fill_value=mean, mask_fill_value=None),
        ToTensorV2()
    ])

    test_transforms = A.Compose([
        A.Normalize(mean, std),
        ToTensorV2()
    ])

    train_data = AlbumentationsCIFAR10Wrapper(root='./data', train=True, 
                                            download=True, transform=train_transforms)

    test_data = AlbumentationsCIFAR10Wrapper(root='./data', train=False,
                                        download=True, transform=test_transforms)
    return train_data, test_data


def draw_training_loss(train_losses, train_acc, test_losses, test_acc):
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")

def unnormalize_tensor(tensor, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(3, 1, 1).to(tensor.device)
    tensor = tensor * std
    tensor = tensor + mean
    return tensor

def unnormalize_numpy(data, mean, std):
    mean = np.array(mean).reshape((3, 1, 1))
    std = np.array(std).reshape((3, 1, 1))
    data = data * std
    data = data + mean
    return data

def unnormalize_minmax(img):
    # img = img / 2 + 0.5  # unnormalize
    # print(np.min(img))
    # print(np.max(img))

    img = img - np.min(img)
    img = img / np.max(img)
    return img

# functions to show an image
def imshow_unnormalized(img):
    # npimg = img.numpy()
    # npimg = unnormalize_minmax(npimg)

    npimg = unnormalize_tensor(img, mean, std).numpy()
    plt.figure(figsize = (4, 4)) # 32x32 pixels = 1.6x1.6 inches at 20 dpi
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    plt.show()

def draw_misclassified_images(misclassified_images, num_images=25):
    fig = plt.figure(figsize=(10,10))
    for i in range(num_images):
        sub = fig.add_subplot(5, 5, i+1)
        plt.imshow(misclassified_images[i][0].cpu().numpy().squeeze(), cmap='gray_r')
        sub.set_title("Pred={}, Act={}".format(str(misclassified_images[i][1].data.cpu().numpy()), str(misclassified_images[i][2].data.cpu().numpy())))
        plt.axis('off')
    plt.tight_layout()

def draw_sample_images(data_loader, num = 4):
    # get some random training images
    dataiter = iter(data_loader)
    images, labels = next(dataiter)

    # show images
    imshow_unnormalized(torchvision.utils.make_grid(images[:num]))
    
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(num)))

def draw_misclassified_images(model, data_loader, device, num = 10):
    processed_count = 0
    for data in data_loader:
        with torch.no_grad():
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            misclassified_indices = (predicted != labels).nonzero(as_tuple=True)[0]

        if processed_count > num:
            break

        misclassified_images = []
        predictions = []
        misclassified_labels = []

        for idx in misclassified_indices:
            if processed_count > num:
                break

            misclassified_images.append(images[idx].cpu().numpy())
            misclassified_labels.append(labels[idx].cpu().numpy())
            predictions.append(predicted[idx].cpu().numpy())
            processed_count += 1

    # Display misclassified images
    fig, axs = plt.subplots(nrows=(int(num/2)), ncols=2, figsize=(7, 14))
    for i in range(int(num/2)):
        ogimg1 = unnormalize_numpy(misclassified_images[i], mean, std)
        ogimg1 = np.transpose(ogimg1, (1, 2, 0))
        axs[i, 0].imshow(ogimg1)
        axs[i, 0].set_title('Predicted: ' + str(classes[predictions[i]]) + ', Actual: ' + str(classes[misclassified_labels[i]]))
        axs[i, 0].axis('off')
        
        second = i+int(num/2)  # the index for the second image, change this as needed
        ogimg2 = unnormalize_numpy(misclassified_images[second], mean, std)
        ogimg2 = np.transpose(ogimg2, (1, 2, 0))
        axs[i, 1].imshow(ogimg2)
        axs[i, 1].set_title('Predicted: ' + str(classes[predictions[second]]) + ', Actual: ' + str(classes[misclassified_labels[second]]))
        axs[i, 1].axis('off')
    plt.show()

def draw_misclassified_with_gradcam_images(model, data_loader, target_layers, device, num = 10):
    processed_count = 0
    for data in data_loader:
        with torch.no_grad():
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            misclassified_indices = (predicted != labels).nonzero(as_tuple=True)[0]

        if processed_count > num:
            break

        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=device.type=="cuda")
        misclassified_images = []
        misclassified_gradcam = []
        predictions = []
        misclassified_labels = []

        for idx in misclassified_indices:
            if processed_count > num:
                break

            misclassified_images.append(images[idx].cpu().numpy())
            misclassified_labels.append(labels[idx].cpu().numpy())
            predictions.append(predicted[idx].cpu().numpy())

            targets = [ClassifierOutputTarget(predicted[idx])]
            grayscale_cam = cam(input_tensor=images[idx].unsqueeze(0), targets=targets)
            grayscale_cam = grayscale_cam[0, :]

            # print(images[idx].shape) # 1 3 32 32
            # print(images[idx].unsqueeze(0).shape) # 3 32 32
            # print(unnormalize(images[idx], mean, std).shape) # 3 32 32

            img = unnormalize_tensor(images[idx], mean, std)
            img = img.detach().cpu().numpy()
            img = np.transpose(img, (1, 2, 0))
            cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
            misclassified_gradcam.append(cam_image)
            processed_count += 1

    # Display misclassified images with Grad-CAM
    fig, axs = plt.subplots(nrows=num, ncols=2, figsize=(7, 20))
    for i in range(num):
        ogimg = unnormalize_numpy(misclassified_images[i], mean, std)
        ogimg = np.transpose(ogimg, (1, 2, 0))
        axs[i, 0].imshow(ogimg)
        axs[i, 0].set_title('Predicted: ' + str(classes[predictions[i]]) + ', Actual: ' + str(classes[misclassified_labels[i]]))
        axs[i, 0].axis('off')
        axs[i, 1].imshow(misclassified_gradcam[i], cmap='jet')
        axs[i, 1].set_title('Grad-CAM')
        axs[i, 1].axis('off')
    plt.show()

def print_featuremaps_hook(self, input, output):
      # Detach one output feature map (one channel)
      for i in range(output.shape[1]):
          feature_map = output[0, i].detach().cpu().numpy()
          
          # Plot the feature map
          plt.figure(figsize=(3, 3))
          plt.imshow(feature_map, cmap='gray')
          plt.show()

def show_batch_images(plt, dataloader, count=12, row = 3, col = 4):
    images, labels = next(iter(dataloader))
    for i in range(count):
        plt.subplot(row, col, i+1)
        plt.tight_layout()
        plt.imshow(images[i].squeeze(0), cmap='gray')
        plt.title(labels[i].item())
        plt.xticks([])
        plt.yticks([])

# visualize the first conv layer filters
# visualize_conv_layer(plt, model_weights[0].cpu(),row = 5, col = 4)
def visualize_conv_layer(plt, layer_weights, x=8, y=8, row = 5, col = 4):
    plt.figure(figsize=(row, col))
    for i, filter in enumerate(layer_weights):
        plt.subplot(x, y, i+1) # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
        plt.imshow(filter[0, :, :].detach(), cmap='gray')
        plt.axis('off')
    plt.show()

def visualize_feature_map(plt, names, layer_weights, x=5, y=4, row = 6, col = 10):
    fig = plt.figure(figsize=(row, col))
    for i in range(len(layer_weights)):
        a = fig.add_subplot(x, y, i+1)
        imgplot = plt.imshow(layer_weights[i])
        a.axis("off")
        a.set_title(names[i].split('(')[0], fontsize=10)
    plt.savefig(str('feature_maps.jpg'), bbox_inches='tight')
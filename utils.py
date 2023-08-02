import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class AlbumentationsCIFAR10Wrapper(Dataset):
    def __init__(self, root='./data', train=True, download=True, transform=None):
        self.data = datasets.CIFAR10(root=root, train=train, download=download)
        self.transform = transform

    def __getitem__(self, idx):
        img, label = self.data[idx]
        img = np.array(img)  # PIL Image to numpy array
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']

        return img, label

    def __len__(self):
        return len(self.data)
    
def get_train_and_test_data(mean = [0.4914, 0.4822, 0.4465], std = [0.247, 0.243, 0.261]):
    train_transforms = A.Compose([
        A.Normalize(mean, std),
        A.PadIfNeeded(min_height=40, min_width=40, always_apply=True),  # 2 x 4 = 8 on each side
        A.RandomCrop(height=32, width=32, always_apply=True),
        A.HorizontalFlip(p=0.5),
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


# Train Phase transformations
# train_transforms = transforms.Compose([
#                                       #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
#                                        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
#                                        transforms.RandomHorizontalFlip(),
#                                        transforms.RandomRotation(15),
#                                        transforms.ToTensor(),
#                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values. 
#                                        ])

# Test Phase transformations
# test_transforms = transforms.Compose([
#                                       #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
#                                        transforms.ToTensor(),
#                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#                                        ])

######################

# import matplotlib.pyplot as plt
# import numpy as np

# # functions to show an image
# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()


# # get some random training images
# dataiter = iter(train_loader)
# images, labels = next(dataiter)
# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# import torchvision
# # show images
# imshow(torchvision.utils.make_grid(images[:4]))
# # print labels
# print(' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

# # # Call the util function to show a batch of images
# # import matplotlib.pyplot as plt
# # fig = plt.figure()
# # show_batch_images(plt, train_loader, 12, 3, 4)


######################


# # Plots
# fig, axs = plt.subplots(2,2,figsize=(15,10))
# axs[0, 0].plot(train_losses)
# axs[0, 0].set_title("Training Loss")
# axs[1, 0].plot(train_acc)
# axs[1, 0].set_title("Training Accuracy")
# axs[0, 1].plot(test_losses)
# axs[0, 1].set_title("Test Loss")
# axs[1, 1].plot(test_acc)
# axs[1, 1].set_title("Test Accuracy")


# # we will save the conv layer weights in this list
# model_weights =[]
# #we will save the 49 conv layers in this list
# conv_layers = []
# # get all the model children as list
# model_children = list(model.children())
# #counter to keep count of the conv layers
# counter = 0
# #append all the conv layers and their respective wights to the list

# model_children = model.children()
# for children in model_children:
#     if type(children) == nn.Sequential:
#         for child in children:
#             if type(child) == nn.Conv2d:
#                 counter += 1
#                 model_weights.append(child.weight)
#                 conv_layers.append(child)

# print(f"Total convolution layers: {counter}")
# print("conv_layers")


# # get some random training images
# dataiter = iter(train_loader)
# images, labels = next(dataiter)
# imshow(torchvision.utils.make_grid(images[:10]))

# image = images[9]
# imshow(image)

# image = image.unsqueeze(0)
# image = image.to(device)

# outputs = []
# names = []
# for layer in conv_layers[0:]:
#     image = layer(image)
#     outputs.append(image)
#     names.append(str(layer))
# print(len(outputs))
# #print feature_maps
# for feature_map in outputs:
#     print(feature_map.shape)


# processed = []
# for feature_map in outputs:
#     feature_map = feature_map.squeeze(0)
#     gray_scale = torch.sum(feature_map,0)
#     # gray_scale = feature_map[0]
#     gray_scale = gray_scale / feature_map.shape[0]
#     processed.append(gray_scale.data.cpu().numpy())
# for fm in processed:
#     print(fm.shape)


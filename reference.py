

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


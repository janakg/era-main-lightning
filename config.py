PATH_DATASETS = os.environ.get("PATH_DATASETS", "./data")
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 512 if AVAIL_GPUS else 64
AVAIL_GPUS

mean = [0.4914, 0.4822, 0.4465]
std = [0.247, 0.243, 0.261]
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
num_classes=10

num_epochs = 2
max_lr_epoch = 1
dropout_value_min = 0.03

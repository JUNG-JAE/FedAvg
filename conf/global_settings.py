# ----------- System parameters ----------- #
LOG_DIR = "./runs"

# ----------- Server parameters ----------- #
NUM_OF_WORKER = 5
TOTAL_ROUND = 20

# ----------- Worker parameters ----------- #
DATA_TYPE = "CIFAR10"
CHANNEL_SIZE = 1 if DATA_TYPE in ["MNIST", "FMNIST"] else 3

if DATA_TYPE == "MNIST":
    LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
elif DATA_TYPE == "FMNIST":
    LABELS = ['Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']
elif DATA_TYPE == "CIFAR10":
    LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

BATCH_SIZE = 64
LEARNING_RATE = 0.001
LEARNING_EPOCH = 10









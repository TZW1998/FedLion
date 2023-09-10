import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset
import argparse
from tensorboardX import SummaryWriter
import os
import shutil
from copy import deepcopy
import torch.nn.functional as F   
import numpy as np
from scipy import io as spio

PHYSICAL_BATCH_SIZE = 1024

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        # x of shape [B, 1, 28, 28]
        x = F.relu(self.conv1(x))  # -> [B, 16, 14, 14]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 16, 13, 13]
        x = F.relu(self.conv2(x))  # -> [B, 32, 5, 5]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 32, 4, 4]
        x = x.view(-1, 32 * 4 * 4)  # -> [B, 512]
        x = F.relu(self.fc1(x))  # -> [B, 32]
        x = self.fc2(x)  # -> [B, 10]
        return x


def clean_dir(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parse command line arguments
parser = argparse.ArgumentParser(description='FMNIST Classification')
parser.add_argument('--batchsize', type=int, default=32, help='Batch size')
parser.add_argument('--lr', type=float, default=0.1, help='Local Learning rate')
parser.add_argument('--glr', type=float, default=1, help='Global Learning rate')
parser.add_argument('--num_nodes', type=int, default=100, help=' Number of clients per round')
parser.add_argument('--local_steps', type=int, default=5, help='Number of local steps')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
parser.add_argument('--total_steps', type=int, default=500, help='Total number of communication steps')
parser.add_argument('--exp_index', type=int, default=0, help='Experiment index')
parser.add_argument('--log_step', type=int, default=50, help='log steps')
args = parser.parse_args()

# Create data loaders
lr = args.lr
glr = args.glr
batch_size = args.batchsize
momentum = args.momentum
weight_decay = args.weight_decay

##### dataset
total_clients = 3579
emnist = spio.loadmat("EMNIST/emnist-digits.mat")

# ------ training images ------ #
train_images = emnist["dataset"][0][0][0][0][0][0]
train_images = train_images.astype(np.float32)
train_images /= 255

# ------ training labels ------ #
train_labels = emnist["dataset"][0][0][0][0][0][1].reshape(240000).tolist()


# ------ test images ------ #
test_images = emnist["dataset"][0][0][1][0][0][0]
test_images = test_images.astype(np.float32)
test_images /= 255

# ------ test labels ------ #
test_labels = emnist["dataset"][0][0][1][0][0][1].reshape(40000).tolist()

# ------ reshape using matlab order ------ #
train_images = train_images.reshape(train_images.shape[0], 1, 28, 28, order="A")
test_images = test_images.reshape(test_images.shape[0], 1, 28, 28, order="A")

# calculate mean and standard deviation ------ #
mean_px = train_images.mean().astype(np.float32)
std_px = train_images.std().astype(np.float32)

# normalize
train_images = (train_images-mean_px)/std_px
test_images = (test_images-mean_px)/std_px

train_dataset = list(map(list, zip(torch.tensor(train_images), train_labels)))
test_dataset = list(map(list, zip(torch.tensor(test_images), test_labels)))

# DataLoader
#train_all_loader = DataLoader(train_dataset, batch_size= 1024, shuffle=False)
test_all_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=False)



train_writers = emnist["dataset"][0][0][0][0][0][2].reshape(240000)
train_writers_set = np.unique(train_writers)


# Create model
model = CNN().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()


# Training loop
com_round = 0
running_loss = 0.0
total = 0

log_dir = 'logs/EMNIST/fedDA_exp{}_bz{}_lr{}_glr{}_mm{}_wd{}_nn{}_ls{}'.format(args.exp_index, batch_size, lr, glr, momentum, weight_decay, args.num_nodes, args.local_steps)
if os.path.exists(log_dir):
    clean_dir(log_dir)
logger = SummaryWriter(logdir= log_dir)
print('Start training with FedDA algorithm, batch size {}, local learning rate {}, momentum {}, weight decay {}, num nodes {}, local steps {}.'.format(batch_size, lr, momentum, weight_decay, args.num_nodes, args.local_steps))


global_weight = deepcopy(model.state_dict()) # be careful if batchnorm is used
global_state = {ni:{"momentum_buffer" : torch.zeros_like(p)} for ni, (_,p) in enumerate(global_weight.items())}

for com_round in range(args.total_steps):

    aggregated_states_m = {ni:{"momentum_buffer" : torch.zeros_like(p)} for ni, (_,p) in enumerate(global_weight.items())}
    aggregated_states_P = {ni:{"momentum_buffer" : torch.zeros_like(p)} for ni, (_,p) in enumerate(global_weight.items())}

    choosed_clients = np.random.choice(train_writers_set,replace=False,size=args.num_nodes)
    for client_ind, client in enumerate(choosed_clients):
        client_writer_indexes = np.where(train_writers==client)[0]
        train_dataset_each_client = DatasetSplit(train_dataset, client_writer_indexes)
                    
        # mini-batch samples
        client_train_loader = torch.utils.data.DataLoader(train_dataset_each_client,
                                            batch_size = min([batch_size, len(train_dataset_each_client)]), shuffle=True)

        # load global weight
        model.load_state_dict(global_weight)
        local_optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0, weight_decay=weight_decay)
        

        # Local training
        num_local_steps = 0

        client_state_m = deepcopy(global_state)
        client_state_P = {ni:{"momentum_buffer" : torch.zeros_like(p)} for ni, (_,p) in enumerate(global_weight.items())}

        while num_local_steps < args.local_steps:
            for i, (images, labels) in enumerate(client_train_loader):
                images = images.to(device)
                labels = labels.to(device)

                # Gradient accumulation if local_batch_size is larger than 1024
                if labels.size(0) <= PHYSICAL_BATCH_SIZE:
                    # Forward pass
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    # Backward 
                    loss.backward()

                    # Compute running loss
                    total += labels.size(0)
                    running_loss += loss.item() * labels.size(0)

                else:
                    if (labels.size(0) % PHYSICAL_BATCH_SIZE) == 0:
                        num_accumulation_steps = (labels.size(0) // PHYSICAL_BATCH_SIZE)  
                    else: 
                        num_accumulation_steps = (labels.size(0) // PHYSICAL_BATCH_SIZE)   + 1
            
                    for now_steps in range(num_accumulation_steps):
                        if now_steps == (num_accumulation_steps - 1):
                            input_var = images[(now_steps * PHYSICAL_BATCH_SIZE):].to(device)
                            target_var = labels[(now_steps * PHYSICAL_BATCH_SIZE):].to(device)
                        else:
                            input_var = images[(now_steps * PHYSICAL_BATCH_SIZE): ((now_steps + 1) * PHYSICAL_BATCH_SIZE)].to(device)
                            target_var = labels[(now_steps * PHYSICAL_BATCH_SIZE): ((now_steps + 1) * PHYSICAL_BATCH_SIZE)].to(device)
                        
                        output = model(input_var)
                        loss = criterion(output, target_var) * target_var.size(0) / labels.size(0)
                        loss.backward()

                        # Compute running loss
                        total += target_var.size(0)
                        running_loss += loss.item() * labels.size(0)

                # update client momentum
                for ni, (_,p) in enumerate(model.named_parameters()):
                    client_state_m[ni]["momentum_buffer"] = momentum * client_state_m[ni]["momentum_buffer"] + (1 - momentum) * p.grad.data
                    client_state_P[ni]["momentum_buffer"] += client_state_m[ni]["momentum_buffer"]

                local_optimizer.step()
                local_optimizer.zero_grad()
                num_local_steps += 1

                if num_local_steps == args.local_steps:
                    break
        
        # all-reduce for model and optimizer state
        for ni, (_, p) in enumerate(aggregated_states_m.items()):
            aggregated_states_m[ni]["momentum_buffer"] += client_state_m[ni]["momentum_buffer"] / args.num_nodes
            aggregated_states_P[ni]["momentum_buffer"] += client_state_P[ni]["momentum_buffer"] / args.num_nodes
            
        
    # update_steps += 1
    # update global_weight and global_state
    global_state = aggregated_states_m
    for ni, (n, p) in enumerate(global_weight.items()):
        global_weight[n] -= glr * lr * aggregated_states_P[ni]["momentum_buffer"]

    # Print loss and accuracy every log_step communication round
    if (args.log_step is None) or (com_round % args.log_step == 0):
        model.load_state_dict(global_weight)
        model.eval()
        with torch.no_grad():
            test_correct = 0
            test_total = 0
            for images, labels in test_all_loader:
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)

                # Compute accuracy
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        # Print test accuracy
        print(''.format())
        model.train()

        print('Communication round: {}, Train Loss: {:.4f}, Test Accuracy: {:.2f}%'.format(com_round, running_loss / total,
                                                                            (test_correct / test_total) * 100))
        logger.add_scalar('running_train_loss', running_loss / total, com_round)
        logger.add_scalar('test_accuracy', (test_correct / test_total) * 100, com_round)

        # logger.add_scalar('running_train_loss_by_update', running_loss / total, update_steps)
        # logger.add_scalar('test_accuracy_by_update', (test_correct / test_total) * 100, update_steps)




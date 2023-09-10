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
from torch.optim import Optimizer
from typing import List, Optional
from torch import Tensor

PHYSICAL_BATCH_SIZE = 1024

class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""
    def __repr__(self):
        return "<required parameter>"

required = _RequiredParameter()

def _use_grad_for_differentiable(func):
    def _use_grad(self, *args, **kwargs):
        prev_grad = torch.is_grad_enabled()
        try:
            torch.set_grad_enabled(self.defaults['differentiable'])
            ret = func(self, *args, **kwargs)
        finally:
            torch.set_grad_enabled(prev_grad)
        return ret
    return _use_grad

class SignSGD(Optimizer):
    def __init__(self, params, lr=required, betas=(0.9, 0.99),
                 weight_decay=0, *,
                 differentiable=False):
        
        momentum = betas[0]
        momentum_interp = betas[1]

        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if momentum_interp < 0.0:
            raise ValueError("Invalid momentum_interp value: {}".format(momentum_interp))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, momentum_interp=momentum_interp,
                        weight_decay=weight_decay, differentiable=differentiable)

        super(SignSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('differentiable', False)

    @_use_grad_for_differentiable
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            error_residuals_list = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            self._update_params(params_with_grad,
                d_p_list,
                momentum_buffer_list,
                weight_decay=group['weight_decay'],
                momentum=group['momentum'],
                lr=group['lr'],
                momentum_interp=group['momentum_interp'])

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss

    def _update_params(self, params: List[Tensor],
            d_p_list: List[Tensor],
            momentum_buffer_list: List[Optional[Tensor]],
            *,
            weight_decay: float,
            momentum: float,
            momentum_interp: float,
            lr: float,
            ):

        for i, param in enumerate(params):
            d_p = d_p_list[i]

            if momentum > 1e-8:
                buf = momentum_buffer_list[i]

                if buf is None:
                    buf = torch.clone(d_p).detach()
                    momentum_buffer_list[i] = buf
                else:
                    grad = buf.mul(momentum).add(d_p, alpha=1-momentum)
                    buf.mul_(momentum_interp).add_(d_p, alpha=1-momentum_interp)
                    d_p = grad
            
            # decouple sign and weight decay
            d_p.sign_()
            if weight_decay != 0:
                d_p.add_(param, alpha=weight_decay)

            # if noise_scale > 1e-8:
            #     d_p.add_(torch.randn_like(d_p), alpha = noise_scale)

            
            param.add_(d_p, alpha=-lr)

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
parser.add_argument('--lr', type=float, default=0.001, help='Local Learning rate')
parser.add_argument('--num_nodes', type=int, default=100, help=' Number of clients per round')
parser.add_argument('--local_steps', type=int, default=20, help='Number of local steps')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Lion')
parser.add_argument('--beta2', type=float, default=0.9, help='beta2 for Lion')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
parser.add_argument('--total_steps', type=int, default=500, help='Total number of communication steps')
parser.add_argument('--exp_index', type=int, default=0, help='Experiment index')
parser.add_argument('--log_step', type=int, default=50, help='log steps')
args = parser.parse_args()

# Create data loaders
lr = args.lr
batch_size = args.batchsize
beta1 = args.beta1
beta2 = args.beta2
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
test_all_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)



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

log_dir = 'logs/EMNIST/avglion_exp{}_bz{}_lr{}_b1{}_b2{}_wd{}_nn{}_ls{}'.format(args.exp_index, batch_size, lr, beta1, beta2, weight_decay, args.num_nodes, args.local_steps)
if os.path.exists(log_dir):
    clean_dir(log_dir)
logger = SummaryWriter(logdir= log_dir)
print('Start training with AVG-Lion algorithm, batch size {}, local learning rate {}, betas ({},{}), weight decay {}, num nodes {}, local steps {}.'.format(batch_size, lr, beta1, beta2, weight_decay, args.num_nodes, args.local_steps))


global_weight = deepcopy(model.state_dict()) # be careful if batchnorm is used
global_state = {ni:{"momentum_buffer" : torch.zeros_like(p)} for ni, (_,p) in enumerate(global_weight.items())}

for com_round in range(args.total_steps):
    # Local SGD
    aggregated_models = {n: torch.zeros_like(p) for n, p in global_weight.items()}
    aggregated_states = {ni:{"momentum_buffer" : torch.zeros_like(p)} for ni, (_,p) in enumerate(global_weight.items())}

    choosed_clients = np.random.choice(train_writers_set,replace=False,size=args.num_nodes)
    for client_ind, client in enumerate(choosed_clients):
        client_writer_indexes = np.where(train_writers==client)[0]
        train_dataset_each_client = DatasetSplit(train_dataset, client_writer_indexes)
                    
        # mini-batch samples
        client_train_loader = torch.utils.data.DataLoader(train_dataset_each_client,
                                            batch_size = min([batch_size, len(train_dataset_each_client)]), shuffle=True)

        # load global weight
        model.load_state_dict(global_weight)
        local_optimizer = SignSGD(model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
        
        # load global optimizer state
        local_optimizer_state = local_optimizer.state_dict()
        local_optimizer_state["state"] = deepcopy(global_state)
        local_optimizer.load_state_dict(local_optimizer_state)

        # Local training
        num_local_steps = 0
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

                local_optimizer.step()
                local_optimizer.zero_grad()
                num_local_steps += 1

                if num_local_steps == args.local_steps:
                    break
        
        # all-reduce for model and optimizer state
        local_weight = model.state_dict()
        local_state = local_optimizer.state_dict()["state"]
        for ni, (n, p) in enumerate(aggregated_models.items()):
            aggregated_models[n] += local_weight[n] / args.num_nodes
            for key in aggregated_states[ni]:
                aggregated_states[ni][key] += local_state[ni][key] / args.num_nodes
        
    # update_steps += 1
    # update global_weight and global_state
    global_weight = aggregated_models
    global_state = aggregated_states

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




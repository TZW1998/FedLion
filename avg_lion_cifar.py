import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import argparse
from tensorboardX import SummaryWriter
import os
import shutil
from copy import deepcopy
import torch.nn.functional as F   
import numpy as np
from scipy import io as spio
import torch.nn.init as init
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


# ============= Neural network : ResNet20 + GN ============= #
def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.in_planes = 16

        # initial
        self.conv0 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0 = nn.GroupNorm(4, 16)  # nn.BatchNorm2d(16)
        self.relu0 = nn.ReLU()

        # block 1
        self.conv11 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn11 = nn.GroupNorm(4, 16)  # nn.BatchNorm2d(16)
        self.relu11 = nn.ReLU()
        self.conv12 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn12 = nn.GroupNorm(4, 16)  # nn.BatchNorm2d(16)
        self.relu12 = nn.ReLU()
        self.shortcut12 = nn.Sequential()

        self.conv13 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn13 = nn.GroupNorm(4, 16)  # nn.BatchNorm2d(16)
        self.relu13 = nn.ReLU()
        self.conv14 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn14 = nn.GroupNorm(4, 16)  # nn.BatchNorm2d(16)
        self.relu14 = nn.ReLU()
        self.shortcut14 = nn.Sequential()

        self.conv15 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn15 = nn.GroupNorm(4, 16)  # nn.BatchNorm2d(16)
        self.relu15 = nn.ReLU()
        self.conv16 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn16 = nn.GroupNorm(4, 16)  # nn.BatchNorm2d(16)
        self.relu16 = nn.ReLU()
        self.shortcut16 = nn.Sequential()

        # block 2
        self.conv21 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn21 = nn.GroupNorm(8, 32)  # nn.BatchNorm2d(32)
        self.relu21 = nn.ReLU()
        self.conv22 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn22 = nn.GroupNorm(8, 32)  # nn.BatchNorm2d(32)
        self.relu22 = nn.ReLU()
        self.shortcut22 = LambdaLayer(lambda x:
                                      F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, 32 // 4, 32 // 4), "constant", 0))

        self.conv23 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn23 = nn.GroupNorm(8, 32)  # nn.BatchNorm2d(32)
        self.relu23 = nn.ReLU()
        self.conv24 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn24 = nn.GroupNorm(8, 32)  # nn.BatchNorm2d(32)
        self.relu24 = nn.ReLU()
        self.shortcut24 = nn.Sequential()

        self.conv25 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn25 = nn.GroupNorm(8, 32)  # nn.BatchNorm2d(32)
        self.relu25 = nn.ReLU()
        self.conv26 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn26 = nn.GroupNorm(8, 32)  # nn.BatchNorm2d(32)
        self.relu26 = nn.ReLU()
        self.shortcut26 = nn.Sequential()

        # block 3
        self.conv31 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn31 = nn.GroupNorm(16, 64)  # nn.BatchNorm2d(64)
        self.relu31 = nn.ReLU()
        self.conv32 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn32 = nn.GroupNorm(16, 64)  # nn.BatchNorm2d(64)
        self.relu32 = nn.ReLU()
        self.shortcut32 = LambdaLayer(lambda x:
                                      F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, 64 // 4, 64 // 4), "constant", 0))

        self.conv33 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn33 = nn.GroupNorm(16, 64)  # nn.BatchNorm2d(64)
        self.relu33 = nn.ReLU()
        self.conv34 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn34 = nn.GroupNorm(16, 64)  # nn.BatchNorm2d(64)
        self.relu34 = nn.ReLU()
        self.shortcut34 = nn.Sequential()

        self.conv35 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn35 = nn.GroupNorm(16, 64)  # nn.BatchNorm2d(64)
        self.relu35 = nn.ReLU()
        self.conv36 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn36 = nn.GroupNorm(16, 64)  # nn.BatchNorm2d(64)
        self.relu36 = nn.ReLU()
        self.shortcut36 = nn.Sequential()

        # final
        self.linear = nn.Linear(64, num_classes)
        self.apply(_weights_init)

    def forward(self, x):
        # initial
        out0 = self.relu0(self.bn0(self.conv0(x)))

        # block 1
        out = self.relu11(self.bn11(self.conv11(out0)))
        out = self.bn12(self.conv12(out))
        out += self.shortcut12(out0)
        out12 = self.relu12(out)

        out = self.relu13(self.bn13(self.conv13(out12)))
        out = self.bn14(self.conv14(out))
        out += self.shortcut14(out12)
        out14 = self.relu14(out)

        out = self.relu15(self.bn15(self.conv15(out14)))
        out = self.bn16(self.conv16(out))
        out += self.shortcut16(out14)
        out16 = self.relu16(out)

        # block 2
        out = self.relu21(self.bn21(self.conv21(out16)))
        out = self.bn22(self.conv22(out))
        out += self.shortcut22(out16)
        out22 = self.relu22(out)

        out = self.relu23(self.bn23(self.conv23(out22)))
        out = self.bn24(self.conv24(out))
        out += self.shortcut24(out22)
        out24 = self.relu24(out)

        out = self.relu25(self.bn25(self.conv25(out24)))
        out = self.bn26(self.conv26(out))
        out += self.shortcut26(out24)
        out26 = self.relu26(out)

        # block 3
        out = self.relu31(self.bn31(self.conv31(out26)))
        out = self.bn32(self.conv32(out))
        out += self.shortcut32(out26)
        out32 = self.relu32(out)

        out = self.relu33(self.bn33(self.conv33(out32)))
        out = self.bn34(self.conv34(out))
        out += self.shortcut34(out32)
        out34 = self.relu34(out)

        out = self.relu35(self.bn35(self.conv35(out34)))
        out = self.bn36(self.conv36(out))
        out += self.shortcut36(out34)
        out36 = self.relu36(out)

        # final
        out = F.avg_pool2d(out36, out36.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

# ============= Database distribution (iid, non-iid) ============= #
def CIFAR10_noniid_diff_label_prob(dataset, num_users, weight_per_label_set):
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(len(dataset))
    labels = np.array([dataset[i][1] for i in range(len(dataset))])

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    classes_each_device = 10
    start_index = [int(len(dataset) / classes_each_device * j) for j in range(classes_each_device)]

    for j in range(classes_each_device):
        for i in range(int(num_users) - 1):
            rand_set = np.arange(start_index[j], start_index[j] + int(
                np.round(len(dataset) / classes_each_device * weight_per_label_set[i][j])))
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand_set]), axis=0)
            start_index[j] = start_index[j] + int(
                np.round(len(dataset) / classes_each_device * weight_per_label_set[i][j]))
        i = int(num_users) - 1
        rand_set = np.arange(start_index[j], int(len(dataset) / classes_each_device * (j + 1)))
        dict_users[i] = np.concatenate((dict_users[i], idxs[rand_set]), axis=0)

    return dict_users

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


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parse command line arguments
parser = argparse.ArgumentParser(description='FMNIST Classification')
parser.add_argument('--batchsize', type=int, default=32, help='Batch size')
parser.add_argument('--lr', type=float, default=0.001, help='Local Learning rate')
parser.add_argument('--num_nodes', type=int, default=10, help=' Number of clients per round')
parser.add_argument('--local_steps', type=int, default=2, help='Number of local steps')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Lion')
parser.add_argument('--beta2', type=float, default=0.9, help='beta2 for Lion')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
parser.add_argument('--total_steps', type=int, default=5000, help='Total number of communication steps')
parser.add_argument('--exp_index', type=int, default=0, help='Experiment index')
parser.add_argument('--log_step', type=int, default=100, help='log steps')
args = parser.parse_args()

# Create data loaders
lr = args.lr
batch_size = args.batchsize
beta1 = args.beta1
beta2 = args.beta2
weight_decay = args.weight_decay

##### dataset
total_clients = 100
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, 4),
    transforms.ToTensor(),
    normalize,
]), download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
    transforms.ToTensor(),
    normalize,
]))
train_all_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1280, shuffle=False, num_workers=0, pin_memory=True)
test_all_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1280, shuffle=False, num_workers=0, pin_memory=True)


weight_per_label_set = np.random.dirichlet(np.ones(10), total_clients)
weight_per_label_set /= np.sum(weight_per_label_set, 0)
node_groups = CIFAR10_noniid_diff_label_prob(train_dataset, total_clients, weight_per_label_set)


# Create model
model = CNN().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()


# Training loop
com_round = 0
running_loss = 0.0
total = 0

log_dir = 'logs/CIFAR/avglion_exp{}_bz{}_lr{}_b1{}_b2{}_wd{}_nn{}_ls{}'.format(args.exp_index, batch_size, lr, beta1, beta2, weight_decay, args.num_nodes, args.local_steps)
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

    choosed_clients = np.random.choice(total_clients,args.num_nodes,replace=False).astype(int)
    for client_ind, client in enumerate(choosed_clients):
        client_writer_indexes = list(node_groups[client])
        train_dataset_each_client = DatasetSplit(train_dataset, client_writer_indexes)
            
        # mini-batch samples
        client_train_loader = torch.utils.data.DataLoader(train_dataset_each_client,
                                                            batch_size=batch_size, shuffle=False,
                                                            sampler=torch.utils.data.sampler.RandomSampler(
                                                                range(len(train_dataset_each_client)),
                                                                replacement=True,
                                                                num_samples=batch_size * args.local_steps),
                                                            num_workers=2, pin_memory=True)

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




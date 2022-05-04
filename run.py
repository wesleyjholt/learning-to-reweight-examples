import torch
from tqdm.notebook import tqdm
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets,transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import time
import copy
import torch.nn.functional as F
import copy
from torch import autograd
import higher
import itertools
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# Defining the network (LeNet-5)  
# from https://github.com/bollakarthikeya/LeNet-5-PyTorch/blob/master/lenet5_gpu.py
class LeNet5(torch.nn.Module):          
    def __init__(self):     
        super(LeNet5, self).__init__()
        # Convolution (In LeNet-5, 32x32 images are given as input. Hence padding of 2 is done below)
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
        self.max_pool_1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
        self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=2) 
        self.fc1 = torch.nn.Linear(16*5*5, 120)   
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))  
        x = self.max_pool_1(x) 
        x = F.relu(self.conv2(x))
        x = self.max_pool_2(x)
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x).squeeze()
        return x

def get_loss_n_accuracy(model, criterion, data_loader, args, num_classes=2):
    """ Returns the loss and total accuracy, per class accuracy on the supplied data loader """
    
    criterion.reduction = 'mean'
    model.eval()                                     
    total_loss, correctly_labeled_samples = 0, 0
    confusion_matrix = torch.zeros(num_classes, num_classes)
    
    # forward-pass to get loss and predictions of the current batch
    for _, (inputs, labels) in enumerate(data_loader):
        inputs, labels = inputs.to(device=args['device'], non_blocking=True),\
                labels.to(device=args['device'], non_blocking=True)
                                            
        # compute the total loss over minibatch
        outputs = model(inputs)
        avg_minibatch_loss = criterion(outputs, labels.type_as(outputs))
        total_loss += avg_minibatch_loss.item()*outputs.shape[0]
                        
        # get num of correctly predicted inputs in the current batch
        pred_labels = (F.sigmoid(outputs) > 0.5).int()
        correctly_labeled_samples += torch.sum(torch.eq(pred_labels.view(-1), labels)).item()
        # fill confusion_matrix
        for t, p in zip(labels.view(-1), pred_labels.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
                                
    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = correctly_labeled_samples / len(data_loader.dataset)
    per_class_accuracy = confusion_matrix.diag() / confusion_matrix.sum(1)
    return avg_loss, (accuracy, per_class_accuracy)

def get_imbalanced_datasets(train_dataset, test_dataset, imbalance=0.995, train_size=5000, meta_size=10):
    # returns an imbalanced mnist dataset of 9 and 4s where imbalance favors 9s
    
    # a balanced test dataset
    test_9_idxs = test_dataset.targets == 9
    test_4_idxs = test_dataset.targets == 4
    test_9_data = test_dataset.data[test_9_idxs][:982]
    test_4_data = test_dataset.data[test_4_idxs][:982] # num of 4 samples 
    test_data = torch.cat((test_9_data, test_4_data))
    test_targets = torch.cat( (torch.ones(len(test_9_data))*1, torch.ones(len(test_4_data))*0 ) )
    test_dataset.data = test_data
    test_dataset.targets = test_targets
    
    # imbalanced training dataset
    n_9s = int(train_size * imbalance)
    n_4s = train_size - n_9s
    train_9_idxs = train_dataset.targets == 9
    train_4_idxs = train_dataset.targets == 4 
    train_9_data = train_dataset.data[train_9_idxs][:n_9s]
    train_4_data = train_dataset.data[train_4_idxs][:n_4s]
    train_data = torch.cat((train_9_data, train_4_data))
    train_targets = torch.cat( (torch.ones(len(train_9_data))*1, torch.ones(len(train_4_data))*0 ) )
    train_dataset.data = train_data
    train_dataset.targets = train_targets
    
    # a balanced meta dataset for weighting samples (which is subset of training dataset)
    # note that we have relabed 9s as 1 and 4s as 0
    meta_dataset = copy.deepcopy(train_dataset)
    meta_9_idxs = meta_dataset.targets == 1
    meta_4_idxs = meta_dataset.targets == 0
    meta_9_data = meta_dataset.data[meta_9_idxs][:(meta_size // 2)]
    meta_4_data = meta_dataset.data[meta_4_idxs][:(meta_size // 2)]
    meta_data = torch.cat((meta_9_data, meta_4_data))
    meta_dataset.data = meta_data
    meta_targets = torch.cat( (torch.ones(len(meta_9_data))*1, torch.ones(len(meta_4_data))*0 ) )
    meta_dataset.targets = meta_targets
    
    return train_dataset, meta_dataset, test_dataset   

args = {'bs':100, 'lr':1e-3, 'n_epochs':150, 'device':'cuda:0'}

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
train_dataset, meta_dataset, test_dataset = get_imbalanced_datasets(train_dataset, test_dataset)

train_loader = DataLoader(train_dataset, batch_size=args['bs'], shuffle=True, num_workers=2, pin_memory=True)
test_loader =  DataLoader(test_dataset, batch_size=args['bs'], shuffle=False, num_workers=2, pin_memory=True)
meta_loader = DataLoader(meta_dataset, batch_size=args['bs'], shuffle=True, pin_memory=True)
meta_loader = itertools.cycle(meta_loader)

model = LeNet5().to(args['device'])
opt = optim.SGD(model.parameters(), lr=args['lr'])
criterion = nn.BCEWithLogitsLoss().to(args['device'])

start_time, end_time = torch.cuda.Event(enable_timing=True),\
                        torch.cuda.Event(enable_timing=True)
writer = SummaryWriter('logs/baseline')
start_time.record()

for ep in tqdm(range(1, args['n_epochs']+1)):
    model.train()
    train_loss, train_acc = 0, 0
    for _, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device=args['device'], non_blocking=True),\
                            labels.to(device=args['device'], non_blocking=True)
        opt.zero_grad()
        outputs = model(inputs)
        minibatch_loss = criterion(outputs, labels.type_as(outputs))
        minibatch_loss.backward()
        opt.step()

        # keep track of epoch loss/accuracy
        train_loss += minibatch_loss.item()*outputs.shape[0]
        pred_labels = (F.sigmoid(outputs) > 0.5).int()
        train_acc += torch.sum(torch.eq(pred_labels, labels)).item()
        
    # inference after epoch
    with torch.no_grad():
        train_loss, train_acc = train_loss/len(train_dataset), train_acc/len(train_dataset)       
        test_loss, (test_acc, test_per_class_acc) = get_loss_n_accuracy(model, criterion, test_loader, args)                                  
        # log/print data
        writer.add_scalar('Test/Loss', test_loss, ep)
        writer.add_scalar('Test/Accuracy', test_acc, ep)
        writer.add_scalar('Training/Loss', train_loss, ep)
        writer.add_scalar('Training/Accuracy', train_acc, ep)
        print(f'|Train/Test Loss: {train_loss:.3f} / {test_loss:.3f}|', end='--')
        print(f'|Train/Test Acc: {train_acc:.3f} / {test_acc:.3f}|', end='\r')    

end_time.record()
torch.cuda.synchronize()
time_elapsed_secs = start_time.elapsed_time(end_time)/10**3
time_elapsed_mins = time_elapsed_secs/60
print(f'Training took {time_elapsed_secs:.2f} seconds / {time_elapsed_mins:.2f} minutes')

model = LeNet5().to(args['device'])
opt = optim.SGD(model.parameters(), lr=args['lr'])
start_time, end_time = torch.cuda.Event(enable_timing=True),\
                        torch.cuda.Event(enable_timing=True)
writer = SummaryWriter('logs/weighted')
start_time.record()

for ep in tqdm(range(1, args['n_epochs']+1)):
    model.train()
    train_loss, train_acc = 0, 0
    for _, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device=args['device'], non_blocking=True),\
                            labels.to(device=args['device'], non_blocking=True)
        
        opt.zero_grad()
        with higher.innerloop_ctx(model, opt) as (meta_model, meta_opt):
            # 1. Update meta model on training data
            meta_train_outputs = meta_model(inputs)
            criterion.reduction = 'none'
            meta_train_loss = criterion(meta_train_outputs, labels.type_as(outputs))
            eps = torch.zeros(meta_train_loss.size(), requires_grad=True, device=args['device'])
            meta_train_loss = torch.sum(eps * meta_train_loss)
            meta_opt.step(meta_train_loss)

            # 2. Compute grads of eps on meta validation data
            meta_inputs, meta_labels =  next(meta_loader)
            meta_inputs, meta_labels = meta_inputs.to(device=args['device'], non_blocking=True),\
                             meta_labels.to(device=args['device'], non_blocking=True)

            meta_val_outputs = meta_model(meta_inputs)
            criterion.reduction = 'mean'
            meta_val_loss = criterion(meta_val_outputs, meta_labels.type_as(outputs))
            eps_grads = torch.autograd.grad(meta_val_loss, eps)[0].detach()

        # 3. Compute weights for current training batch
        w_tilde = torch.clamp(-eps_grads, min=0)
        l1_norm = torch.sum(w_tilde)
        if l1_norm != 0:
            w = w_tilde / l1_norm
        else:
            w = w_tilde

        # 4. Train model on weighted batch
        outputs = model(inputs)
        criterion.reduction = 'none'
        minibatch_loss = criterion(outputs, labels.type_as(outputs))
        minibatch_loss = torch.sum(w * minibatch_loss)
        minibatch_loss.backward()
        opt.step()

        # keep track of epoch loss/accuracy
        train_loss += minibatch_loss.item()*outputs.shape[0]
        pred_labels = (F.sigmoid(outputs) > 0.5).int()
        train_acc += torch.sum(torch.eq(pred_labels, labels)).item()

    # inference after epoch
    with torch.no_grad():
        train_loss, train_acc = train_loss/len(train_dataset), train_acc/len(train_dataset)       
        test_loss, (test_acc, test_per_class_acc) = get_loss_n_accuracy(model, criterion, test_loader, args)                                  
        # log/print data
        writer.add_scalar('Test/Loss', test_loss, ep)
        writer.add_scalar('Test/Accuracy', test_acc, ep)
        writer.add_scalar('Training/Loss', train_loss, ep)
        writer.add_scalar('Training/Accuracy', train_acc, ep)
        print(f'|Train/Test Loss: {train_loss:.3f} / {test_loss:.3f}|', end='--')
        print(f'|Train/Test Acc: {train_acc:.3f} / {test_acc:.3f}|', end='\r')    

end_time.record()
torch.cuda.synchronize()
time_elapsed_secs = start_time.elapsed_time(end_time)/10**3
time_elapsed_mins = time_elapsed_secs/60
print(f'Training took {time_elapsed_secs:.2f} seconds / {time_elapsed_mins:.2f} minutes')
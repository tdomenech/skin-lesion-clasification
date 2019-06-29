from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import my_resnet
from tqdm import tqdm
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score

w_loss_n = []

def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses #vector amb tants zeros com nombre de classes
    for item in images: #conta les imatges que hi ha a cada classe
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses #vector amb tants 0. com nombre de classes
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i]) #total/el nombre d'imatges per aquella classe
    weight = [0] * len(images) #defineix weight com un vector de zeros de longitud nombre d'imatges
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]] #aqui fa, per cada imatge, li assigna el weight de la classe
    return weight
#
def make_weights_for_loss_function(images, nclasses):
    count = [0] * nclasses  # vector amb tants zeros com nombre de classes
    for item in images:  # conta les imatges que hi ha a cada classe
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses  # vector amb tants 0. com nombre de classes
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])  # total/el nombre d'imatges per aquella classe
    return weight_per_class

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),         #multiples de 90, 90, 180, 270
        transforms.RandomRotation(90),
        transforms.RandomRotation(270),
        transforms.ToTensor()
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ]),
} #per normalitzar les imatges.


data_dir = '/imatge/tdomenech/work/data/ISIC-images/'

escriptura_loss_train = 'train_loss.txt'
escriptura_loss_val = 'val_loss.txt'
escriptura_acc_train = 'train_acc.txt'
escriptura_acc_val = 'val_acc.txt'


train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train']) #mirar si ho fa random
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val'])

print('Classes to index', train_dataset.class_to_idx)

weights_train = make_weights_for_balanced_classes(train_dataset.imgs, len(train_dataset.classes)) #crees els weights per les images i les classes
weights_train = torch.DoubleTensor(weights_train) #n-dimensional array, és més fàcil per operar que una matriu
sampler_train = torch.utils.data.sampler.WeightedRandomSampler(weights_train, len(weights_train)) #agafa mostres amb la probabilitat de weights_train

weights_val = make_weights_for_balanced_classes(val_dataset.imgs, len(val_dataset.classes)) #igual pero per val
weights_val = torch.DoubleTensor(weights_val)
sampler_val = torch.utils.data.sampler.WeightedRandomSampler(weights_val, len(weights_val))

w_loss = make_weights_for_loss_function(train_dataset.imgs, len(train_dataset.classes))
w_loss = torch.FloatTensor(w_loss)

#classes no balancejades:
#train_dataloader = DataLoader(train_dataset, batch_size=16, num_workers=4, shuffle=True)
#val_dataloader = DataLoader(val_dataset, batch_size=16, num_workers=4, shuffle=True)

#classes balancejades:
train_dataloader = DataLoader(train_dataset, batch_size=36, num_workers=4, sampler=sampler_train) #combina un dataset i un sampler. Agafa dades del train set, la mida del batch, el nombre de subprocessos per carregar les dades i el sampler (quines dades s'han d'agafar)
val_dataloader = DataLoader(val_dataset, batch_size=36, num_workers=4, sampler=sampler_val)


dataloaders = {'train': train_dataloader, 'val': val_dataloader}

dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
class_names = train_dataset.classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=5)
#f = open( 'fitxer_loss_acc_100epochs.txt', 'a+' ) #fitxer de sortida de loss i accuracy

def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()


    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 1000.0
    num_epoch_loss = 0
    num_epoch_acc = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                if scheduler is not None:
                    scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            balanced_acc = 0.0
            data_acc = torch.tensor([], dtype=torch.long, device='cuda:0')
            preds_acc = torch.tensor([], dtype=torch.long, device='cuda:0')

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase], total=len(dataloaders[phase])):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()


                # statistics
                running_loss += loss.item() * inputs.size(0)

                running_corrects += torch.sum(preds == labels.data) #casos que les prediccions son iguals que la data

                #per la confusion matrix el que li he d'entrar labels i preds, però concatenant:
                data_acc = torch.cat((data_acc, labels.data), 0)
                preds_acc = torch.cat((preds_acc, preds), 0)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            #Calculem l'epoch acc de manera diferent:
            epoch_balanced_acc = balanced_accuracy_score(data_acc.cpu(), preds_acc.cpu())
            print()

            print('{} Loss: {:.4f} Acc: {:.4f} Balanced_Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc, epoch_balanced_acc))

            #f.write('Epoch {}/{}'.format(epoch, num_epochs - 1) + '\n' + 'Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc) + '\n')
            if phase == 'train':
                with open(escriptura_loss_train, 'a+') as f:
                    f.write('{}\n'.format(epoch_loss))
                with open(escriptura_acc_train, 'a+') as f:
                    f.write('{}\n'.format(epoch_balanced_acc))

            if phase == 'val':
                # copy loss,acc val in a file
                with open(escriptura_loss_val, 'a+') as f:
                    f.write('{}\n'.format(epoch_loss))
                with open(escriptura_acc_val, 'a+') as f:
                    f.write('{}\n'.format(epoch_balanced_acc))



            # deep copy the model

            # cambiar per loss de validacio (mínima, no maxima)
            if phase == 'val' and epoch_balanced_acc > best_acc:
                best_acc = epoch_balanced_acc
                num_epoch_acc = num_epochs - 1

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                num_epoch_loss = num_epochs - 1
                best_model_wts = copy.deepcopy(model.state_dict())

        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss), 'in epoch', num_epoch_loss)
    print('Best train Acc: {:4f}'.format(best_acc), 'in epoch', num_epoch_acc)
    # load best model weights
    model.load_state_dict(best_model_wts)
    #f.close()#tancar el fitxer
    return model

model_ft = my_resnet.resnet101(pretrained=True) #carregar un model amb uns weights que ja han fet un train, es descarreguen els weights
num_ftrs = model_ft.fc.in_features #numero de features
model_ft.fc = nn.Linear(num_ftrs, 7) #aplica transformacio linear y=Atx+b, nou model de 2 classes, nosaltres ho haurem de canviar a 7 (l'imagen net, per defecte son 1000)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss(weight=(w_loss.cuda())) #combina nn.LogSoftmax() i nn.NLLLoss() en una sola classe. És útil quan tenim un nombre determinat de classes. És la manera de calcular la loss

# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(model_ft.parameters(), lr=1e-6) #s'optimitza amb el mètode d'Adam, amb aquest learning rate que normalment és el que s'ha de provar per veure quin funciona

# Decay LR by a factor of 0.1 every 7 epochs
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)
exp_lr_scheduler = None

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=40) #s'entrena el model amb la resnet18,

torch.save(model_ft, 'output_model_2018_101.pth')
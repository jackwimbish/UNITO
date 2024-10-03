import torch
import os
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch.optim as optim
from UNITO_Model import UNITO
from Utils_Train import *

def train(gate, learning_rate, device, batch_size, epoches, n_worker, dest):
  """
  training UNITO using the tuned hyperparameters with BCEWithLogitsLoss and Adam optimizer
  args:
    gate: current gate
    learning_rate: step strength used to train the model
    device: whether use GPU or not
    batch_size: number of samples processed together in one time
    epoches: number of epoches the model will be trained
    n_worker: number of worker for pytorch setting
  """

  # process data
  path_train = pd.read_csv(f'{dest}/Data/Data_{gate}/train/subj.csv')

  train_transforms = A.Compose(
      [
        ToTensorV2(),
      ],
  )

  model = UNITO(in_channels = 1, out_channels = 1).to(device)
  loss_fn = nn.BCEWithLogitsLoss()
  optimizer = optim.Adam(model.parameters(), lr = learning_rate)

  train_ds = dataset(path_train, train_transforms)
  train_loader = DataLoader(train_ds, batch_size = batch_size, num_workers = n_worker, pin_memory = True)

  # train
  for epoch in range(epoches):
    loss = train_epoch(train_loader, model, optimizer, loss_fn, device)

  PATH = os.path.join(f'{dest}/model/', gate+'_model.pt')
  torch.save(model.state_dict(), PATH)


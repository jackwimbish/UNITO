import torch
from torch.utils.data import DataLoader
from Dataset import dataset

def get_loaders(path_train, path_test, batch_size, train_transform, test_transform, num_workers = 2, pin_memory = False):
  """
  Prepare data loader for pytorch model
  args:
    path_train: path of the subject list for training data
    path_test: path of the subject list for testing data
    batch_size: number of samples processed together in one time
    train_transform: data augmentation parameter for training
    test_transform: data augmentation parameter for testing
    num_workers: number of worker for pytorch setting 
    pin_memory: helps data allocation for faster loading
  """
  train_ds = dataset(path_train, train_transform)
  train_loader = DataLoader(train_ds, batch_size = batch_size, num_workers = num_workers, pin_memory = pin_memory, shuffle = True)
  test_ds = dataset(path_test, test_transform)
  test_loader = DataLoader(test_ds, batch_size = batch_size, num_workers = num_workers, pin_memory = pin_memory, shuffle = True)
  return train_loader, test_loader

def train_epoch(loader, model, optimizer, loss_fn, device):
  """
  Perform one epoch of training
  args:
    loader: prepared data loader
    model: UNITO model
    optimizer: selected optimizer, default is Adam
    loss_fn: selected loss function, default is BCEWithLogitsLoss
    device: whether use GPU
  """
  loss_total = 0
  for batch_idx, (data, target, subj) in enumerate(loader):
    data = data.type(torch.float32)
    data = data.to(device = device)
    target = target.type(torch.float32)
    target = target.unsqueeze(1).to(device = device)

    # forward
    predictions = model(data)
    loss = loss_fn(predictions, target)
    loss_total += loss.item()     

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  return loss_total/(batch_idx+1)

def check_accuracy(loader, model, device="mps"):
  """
  Calculating the accuracy and dice score for assigned data loader
  args:
    loader: prepared data loader
    model: UNITO model
    device: whether use GPU
  """
  num_correct = 0
  num_pixels = 0
  dice_score = 0
  model.eval()

  with torch.no_grad():
    for x,y, subj in loader:
      x = x.type(torch.float32)
      x = x.to(device)
      y = y.type(torch.float32)
      y = y.to(device).unsqueeze(1)
      preds = torch.sigmoid(model(x))
      preds = (preds > 0.5).float()
      num_correct += (preds == y).sum()
      num_pixels += torch.numel(preds)
      dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
  model.train()

  return num_correct/num_pixels*100, dice_score/len(loader)

def predict_visualization(loader, model, device="mps"):
  """
  Predict using the trained model, output corresponding cytometric parameters with the predicted mask for visualization
  args:
    loader: prepared data loader
    model: UNITO model
    device: whether use GPU
  """
  model.eval()
  preds_list = []
  y_list = []
  x_list = []
  subj_list = []
  for idx, (x,y,subj) in enumerate(loader):
    x = x.type(torch.float32)
    x = x.to(device=device)
    with torch.no_grad():
      preds = torch.sigmoid(model(x))
      preds = (preds > 0.5).float()
    preds_list.append(preds)
    y_list.append(y.unsqueeze(1))
    x_list.append(x.unsqueeze(1))
    subj_list.append(subj)

  return preds_list, y_list, x_list, subj_list

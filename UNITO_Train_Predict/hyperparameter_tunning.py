import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch.optim as optim
from UNITO_Model import UNITO
from Utils_Train import get_loaders, train_epoch, check_accuracy

def tune(gate, hyperparameter_set, device, epoches, n_worker, dest):
    """
    Using one of the cross validation dataset to find the optimal hyperparameter, including batch size and learning rate
    args:
        gate: current gate
        hyperparameter_set: combinations of different hyperparameter setting
        device: whether use GPU or not
        epoches: number of epoches the model will be trained
        n_worker: number of worker for pytorch setting
        dest: destination folder for output
    """
    min_dice = 0
    best_lr = 0.001
    best_bs = 8

    # load subject list
    path = pd.read_csv(f'{dest}/Data/Data_{gate}/train/subj.csv')
    cutoff = int(path.shape[0]//8*7)
    path_train = path.iloc[:cutoff]
    path_test = path.iloc[cutoff:]

    # iterate through hyperparameter sets
    for i, hyperparameter in enumerate(hyperparameter_set):
        learning_rate = hyperparameter[0]
        batch_size = hyperparameter[1]

        train_transforms = A.Compose(
            [
            ToTensorV2(),
            ],
        )
        test_transforms = A.Compose(
            [
            ToTensorV2(),
            ],
        )

        # initialize model
        model = UNITO(in_channels = 1, out_channels = 1).to(device)
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr = learning_rate)

        train_loader, test_loader = get_loaders(path_train, path_test, batch_size, train_transforms, test_transforms, num_workers = n_worker, pin_memory = False)

        for epoch in range(epoches):
            loss = train_epoch(train_loader, model, optimizer, loss_fn, device)

            # check accuracy - test
            test_accuracy, test_dice_score = check_accuracy(test_loader, model, device)

            # record the best hyperparameter
            if test_dice_score.cpu() > min_dice:
                min_dice = test_dice_score.cpu()
                best_lr = learning_rate
                best_bs = batch_size

    return best_lr, best_bs
    

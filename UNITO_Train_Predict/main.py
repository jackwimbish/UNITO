from hyperparameter_tunning import tune
from Train import train
from Validation_Recon_Plot_Single import plot_all
from Data_Preprocessing import process_table, train_test_val_split
from Predict import UNITO_gating, evaluation
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# For reproducibility
import torch
torch.manual_seed(0)
import random
random.seed(0)
import numpy as np
np.random.seed(0)

# setting gates
gating = pd.read_csv('./gating_structure.csv')
gate_pre_list = list(gating.Parent_Gate)
gate_pre_list[0] = None # the first gate does not have parent gate
gate_list = list(gating.Gate)
x_axis_list = list(gating.X_axis)
y_axis_list = list(gating.Y_axis)
# path is "./prediction" if there is a parent gate, "./Raw_Data_pred" if there is no parent (i.e. parent is root)
path2_lastgate_pred_list = ['./prediction/' if not pd.isna(g) else './Raw_Data_pred/' for g in gate_pre_list]

# hyperparameter
# device = 'cuda' if torch.cuda.is_available() else 'cpu' 
device = 'mps'
n_worker = 0
epoches = 1000
tuning_epochs = 100
convex = True

hyperparameter_set = [
                      [1e-3, 8],
                      [1e-4, 8],
                      [1e-3, 16],
                      [1e-4, 16]
                      ]

# define path
dest = '.' # change depending on your needs
save_data_img_path = f'{dest}/Data'
save_figure_path = f'{dest}/figures'
save_model_path = f'{dest}/model'
save_prediction_path = f'{dest}/prediction'

# make directory
if not os.path.exists(save_data_img_path):
    os.mkdir(save_data_img_path)
if not os.path.exists(save_figure_path):
    os.mkdir(save_figure_path)
if not os.path.exists(save_model_path):
    os.mkdir(save_model_path)
if not os.path.exists(save_prediction_path):
    os.mkdir(save_prediction_path)
    
###########################################################
# UNITO
###########################################################
hyperparameter_df = pd.DataFrame(columns = ['gate','learning_rate','batch_size'])
has_hyperparameters = False

for i, (gate_pre, gate, x_axis, y_axis, path_raw) in enumerate(zip(gate_pre_list, gate_list, x_axis_list, y_axis_list, path2_lastgate_pred_list)):

    print(f"start UNITO for {gate}")

    # 1. preprocess training data
    train_path = './Raw_Data_train'
    process_table(x_axis, y_axis, gate_pre, gate, train_path, convex, seq = (gate_pre!=None), dest = dest)
    train_test_val_split(gate, train_path, dest, "train")

    # 2. train
    if not has_hyperparameters:
        print("getting hyperparameters")
        if os.path.exists('./hyperparameter_tunning.csv'):
            hyperparameter_df = pd.read_csv('./hyperparameter_tunning.csv')
            best_lr = float(hyperparameter_df.iloc[0]['learning_rate'])
            best_bs = int(hyperparameter_df.iloc[0]['batch_size'])
        else:
            best_lr, best_bs = tune(gate, hyperparameter_set, device, tuning_epochs, n_worker, dest)
            hyperparameter_df.loc[len(hyperparameter_df)] = [gate, best_lr, best_bs]
            print(f"got hyperparameters: LR:{best_lr}, BS:{best_bs}")
            print(hyperparameter_df)
            hyperparameter_df.to_csv('./hyperparameter_tunning.csv')
        has_hyperparameters = True
    train(gate, best_lr, device, best_bs, epoches, n_worker, dest)

    # 3. preprocess training data
    print(f"Start prediction for {gate}")
    pred_path = './Raw_Data_pred'
    process_table(x_axis, y_axis, gate_pre, gate, pred_path, convex, seq = (gate_pre!=None), dest = dest)
    train_test_val_split(gate, pred_path, dest, 'pred')

    # 4. predict
    model_path = f'{dest}/model/{gate}_model.pt'
    data_df_pred = UNITO_gating(model_path, x_axis, y_axis, gate, path_raw, n_worker, device, save_prediction_path, dest, seq = (gate_pre!=None), gate_pre=gate_pre)

    # 5. Evaluation
    accuracy, recall, precision, f1 = evaluation(data_df_pred, gate)
    print(f"{gate}: accuracy:{accuracy}, recall:{recall}, precition:{precision}, f1 score:{f1}")

    # 6. Plot gating results
    plot_all(gate_pre, gate, x_axis, y_axis, path_raw, save_figure_path)
    print("All UNITO prediction visualization saved")

print("Seqential autogating prediction finished")


hyperparameter_df.to_csv('./hyperparameter_tunning.csv')

from Data_Preprocessing import process_table, prepare_subj
from Predict import UNITO_gating, evaluation
from Utils_Plot import plot_all
import os
import warnings
warnings.filterwarnings("ignore")

# setting gates
gate_pre_list = [None, 'Lymphocyte']
gate_list = ['Lymphocyte', 'Singlet']
x_axis_list = ['FSC_A', 'FSC_W']
y_axis_list = ['SSC_A', 'SSC_W']
path2_lastgate_pred_list = ['./Raw_Data/', './prediction/']

# define model parameters
device = 'cpu' # can change to 'cuda' or 'mps', but prediction with cpu is sufficient and works for all computers
n_worker = 0

# define path
dest = '.' # change depending on your needs
save_data_img_path = f'{dest}/Data_image'
save_figure_path = f'{dest}/figures'
save_prediction_path = f'{dest}/prediction'

# make directory
if not os.path.exists(save_data_img_path):
    os.mkdir(save_data_img_path)
if not os.path.exists(save_figure_path):
    os.mkdir(save_figure_path)
if not os.path.exists(save_prediction_path):
    os.mkdir(save_prediction_path)

# ###########################################################
# # UNITO
# ###########################################################

for i, (gate_pre, gate, x_axis, y_axis, path_raw) in enumerate(zip(gate_pre_list, gate_list, x_axis_list, y_axis_list, path2_lastgate_pred_list)):
    
    print(f"Starting UNITO for {gate}")

    # 1. preprocess data
    process_table(x_axis, y_axis, gate_pre, gate, path_raw, seq = (gate_pre!=None))
    prepare_subj(gate)

    # predict
    model_path = f'./Model/{gate}_model.pt'
    data_df_pred = UNITO_gating(model_path, x_axis, y_axis, gate, path_raw, n_worker, device, save_prediction_path, seq = (gate_pre!=None), gate_pre=gate_pre)

    # Evaluation
    accuracy, recall, precision, f1 = evaluation(data_df_pred, gate)
    print(f"{gate}: accuracy:{accuracy}, recall:{recall}, precition:{precision}, f1 score:{f1}")

    # Plot gating results
    plot_all(gate_pre, gate, x_axis, y_axis, path_raw, save_figure_path)
    print("All UNITO prediction visualization saved")

print("Seqential autogating prediction finished")

    

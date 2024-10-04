# Cytometry_PreGating
With the advancement in cytometric technology, there are a large number of cytometric data analysis tools available, but the problem of biological variance across subjects still have not been addressed in other softwares. This pipeline provides an effective way to address this problem and clean the cytometric data in excluding doublets and debris. The pipeline provides two detecting modes, which is single gate prediction and two gates sequential prediction.

## Usage
1. Put the raw tabular data into the project folder with name 'Raw_Data'
2. If running the single gate prediction pipeline, run the Single_Gate.py with arguments of two measurments and the gate name in the csv file. If running the sequential gate prediction, run the Sequential_Gate.py with arguements of two sets of measurements and gate respectively. 

Single prediction example command:
python3 Single_Gate.py --g gate1_ir --x Ir191Di___191Ir_DNA1 --y Event_length --d mps

Sequential prediction example command:
python3 Sequential_Gate.py --g1 gate1_ir --x1 Ir191Di___191Ir_DNA1 --y1 Event_length --g2 gate2_cd45 --x2 Ir193Di___193Ir_DNA2 --y2 Y89Di___89Y_CD45 --d mps

3. To visualize the segmentation of the singlets in the data, we provide a Validation_Recon_Plot.py to reconstruct the predicted label for each cell to a binary map.

Single reconstruction prediction exmaple command:
python3 Validation_Recon_Plot_Single.py --g gate2_cd45 --x Ir193Di___193Ir_DNA2 --y Y89Di___89Y_CD45

Sequential reconstruction prediction exmaple command:
python3 Validation_Recon_Plot_Sequential.py --g1 gate1_ir --x1 Ir191Di___191Ir_DNA1 --y1 Event_length --g2 gate2_cd45 --x2 Ir193Di___193Ir_DNA2 --y2 Y89Di___89Y_CD45
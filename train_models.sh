#Train model with the selected hyperparameters
python3 "peptide_coefficient_predictor.py" \
         '--peptide_file' "preprocess_datasets/preprocessed_datasets/2019_guo_nci60_onehot_encoded_peptide_quants.tsv" \
         '--n_runs' "120" '--seq_length' "60" \
         '--output_file' "2019_guo_nci60/2019_guo_nci60" \
         '--filter_size' '3' \
         '--n_filters' '10' \
         '--n_layers' '4' \
         '--n_nodes' '40' \
         '--dropout' '0.25' \
         '--learning_rate' '0.001' \
         '--batch' '1000' \
         '--random_run' '1'
         
         
#Train model with the selected hyperparameters
python3 "peptide_coefficient_predictor.py" \
         '--peptide_file' "preprocess_datasets/preprocessed_datasets/2015_slevsek_yeast_onehot_encoded_peptide_quants.tsv" \
         '--n_runs' "18" '--seq_length' "60" \
         '--output_file' "2015_slevsek_yeast/2015_slevsek_yeast" \
         '--filter_size' '5' \
         '--n_filters' '40' \
         '--n_layers' '3' \
         '--n_nodes' '20' \
         '--dropout' '0.25' \
         '--learning_rate' '0.005' \
         '--batch' '500' \
         '--random_run' '0'
     

#Train model with the selected hyperparameters
python3 "peptide_coefficient_predictor.py" \
         '--peptide_file' "preprocess_datasets/preprocessed_datasets/2020_thomas_ovarian_onehot_encoded_peptide_quants.tsv" \
         '--n_runs' "103" '--seq_length' "60" \
         '--output_file' "2020_thomas_ovarian/2020_thomas_ovarian" \
         '--filter_size' '6' \
         '--n_filters' '5' \
         '--n_layers' '1' \
         '--n_nodes' '20' \
         '--dropout' '0.0' \
         '--learning_rate' '0.005' \
         '--batch' '500' \
         '--random_run' '0'
        
    
#Train model with the selected hyperparameters
python3 "peptide_coefficient_predictor.py" \
         '--peptide_file' "preprocess_datasets/preprocessed_datasets/CPTAC_S016_COLORECTAL_onehot_encoded_peptide_quants.tsv" \
         '--n_runs' "95" '--seq_length' "60" \
         '--output_file' "CPTAC_S016_COLORECTAL/CPTAC_S016_COLORECTAL" \
         '--filter_size' '5' \
         '--n_filters' '5' \
         '--n_layers' '3' \
         '--n_nodes' '80' \
         '--dropout' '0.5' \
         '--learning_rate' '0.005' \
         '--batch' '500' \
         '--random_run' '0'
    
        
#Train model with the selected hyperparameters
python3 "peptide_coefficient_predictor.py" \
         '--peptide_file' "preprocess_datasets/preprocessed_datasets/CPTAC_S019_COLON_onehot_encoded_peptide_quants.tsv" \
         '--n_runs' "30" '--seq_length' "60" \
         '--output_file' "CPTAC_S019_COLON/CPTAC_S019_COLON" \
         '--filter_size' '7' \
         '--n_filters' '40' \
         '--n_layers' '2' \
         '--n_nodes' '80' \
         '--dropout' '0.25' \
         '--learning_rate' '0.0001' \
         '--batch' '250' \
         '--random_run' '0'


python3 "peptide_coefficient_predictor.py" \
         '--peptide_file' "preprocess_datasets/preprocessed_datasets/CPTAC_S037_COLON_onehot_encoded_peptide_quants.tsv" \
         '--n_runs' "100" '--seq_length' "60" \
         '--output_file' "CPTAC_S037_COLON/CPTAC_S037_COLON" \
         '--filter_size' '5' \
         '--n_filters' '40' \
         '--n_layers' '2' \
         '--n_nodes' '80' \
         '--dropout' '0.5' \
         '--learning_rate' '0.005' \
         '--batch' '250' \
         '--random_run' '0'
         
#Train model with the selected hyperparameters
python3 "peptide_coefficient_predictor.py" \
         '--peptide_file' "preprocess_datasets/preprocessed_datasets/CPTAC_S047_GBM_onehot_encoded_peptide_quants.tsv" \
         '--n_runs' "226" '--seq_length' "60" \
         '--output_file' "CPTAC_S047_GBM/CPTAC_S047_GBM" \
         '--filter_size' '7' \
         '--n_filters' '5' \
         '--n_layers' '4' \
         '--n_nodes' '80' \
         '--dropout' '0.25' \
         '--learning_rate' '0.005' \
         '--batch' '500' \
         '--random_run' '0'

#Train model with the selected hyperparameters          
python3 "peptide_coefficient_predictor.py" \
         '--peptide_file' "preprocess_datasets/preprocessed_datasets/CPTAC_S048_GBM_onehot_encoded_peptide_quants.tsv" \
         '--n_runs' "109" '--seq_length' "60" \
         '--output_file' "CPTAC_S048_GBM/CPTAC_S048_GBM" \
         '--filter_size' '7' \
         '--n_filters' '20' \
         '--n_layers' '3' \
         '--n_nodes' '40' \
         '--dropout' '0.25' \
         '--learning_rate' '0.005' \
         '--batch' '250' \
         '--random_run' '0'
       
#Train model with the selected hyperparameters      
python3 "peptide_coefficient_predictor.py" \
         '--peptide_file' "preprocess_datasets/preprocessed_datasets/CPTAC_S054_HNSCC_onehot_encoded_peptide_quants.tsv" \
         '--n_runs' "191" '--seq_length' "60" \
         '--output_file' "CPTAC_S054_HNSCC/CPTAC_S054_HNSCC" \
         '--filter_size' '7' \
         '--n_filters' '20' \
         '--n_layers' '3' \
         '--n_nodes' '40' \
         '--dropout' '0.25' \
         '--learning_rate' '0.001' \
         '--batch' '250' \
         '--random_run' '0' 
                       
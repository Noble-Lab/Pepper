
#Train model with the selected hyperparameters
python3 "train_neural_network_with_label_noise.py" \
         '--peptide_file' "../preprocess_datasets/preprocessed_datasets/2019_guo_nci60_onehot_encoded_peptide_quants.tsv" \
         '--n_runs' "120" '--seq_length' "60" \
         '--filter_size' '3' \
         '--n_filters' '10' \
         '--n_layers' '4' \
         '--n_nodes' '40' \
         '--dropout' '0.25' \
         '--learning_rate' '0.001' \
         '--batch' '1000' \
         '--label_noise_ratio' '0.4'  
                      
            
#Train robust model with the selected hyperparameters
python3 "train_robust_neural_network_with_label_noise.py" \
         '--peptide_file' "../preprocess_datasets/preprocessed_datasets/2019_guo_nci60_onehot_encoded_peptide_quants.tsv" \
         '--n_runs' "120" '--seq_length' "60" \
         '--filter_size' '3' \
         '--n_filters' '10' \
         '--n_layers' '4' \
         '--n_nodes' '40' \
         '--dropout' '0.25' \
         '--learning_rate' '0.001' \
         '--batch' '1000' \
         '--label_noise_ratio' '0.6'  
                      

#Tune lambda hyperparameter ofor robust model          
python3 "tune_robust_neural_network_lambda.py" \
         '--peptide_file' "../preprocess_datasets/preprocessed_datasets/2019_guo_nci60_onehot_encoded_peptide_quants.tsv" \
         '--n_runs' "120" '--seq_length' "60" \
         '--filter_size' '3' \
         '--n_filters' '10' \
         '--n_layers' '4' \
         '--n_nodes' '40' \
         '--dropout' '0.25' \
         '--learning_rate' '0.001' \
         '--batch' '1000' \
         '--label_noise_ratio' '0.8' \
         '--random_run' '0' \
         '--lam' '1'
  


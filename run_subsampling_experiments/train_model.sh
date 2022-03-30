#Train model with the selected hyperparameters
python3 "peptide_coefficient_predictor_run_subsampling.py" \
         '--peptide_file' "../preprocess_datasets/preprocessed_datasets/2019_guo_nci60_onehot_encoded_peptide_quants.tsv" \
         '--n_runs' "120" '--seq_length' "60" \
         '--filter_size' '3' \
         '--n_filters' '10' \
         '--n_layers' '4' \ 
         '--n_nodes' '40' \
         '--dropout' '0.25' \
         '--learning_rate' '0.001' \
         '--batch' '1000' \
         '--subsample_size' '2'
         '--random_run' '0' \
         
                      
             



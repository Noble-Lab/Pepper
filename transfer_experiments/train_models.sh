#Train joint CPTAC data model
python3 "../peptide_coefficient_predictor.py" \
         '--peptide_file' "../preprocess_datasets/preprocessed_datasets/all_CPTAC_Lumos_TMT11_datasets_except_CPTAC_S051_BRCA_joined_onehot_encoded_peptide_quants.tsv" \
         '--n_runs' "765" '--seq_length' "60" \
         '--output_file' "all_TMT11_lumos_datasets/all_TMT11_lumos_datasets" \
         '--filter_size' '7' \
         '--n_filters' '5' \
         '--n_layers' '4' \
         '--n_nodes' '80' \
         '--dropout' '0.25' \
         '--learning_rate' '0.005' \
         '--batch' '500' \
         '--random_run' '0'


#Run transfer experiment
python3 "peptide_coefficient_predictor_transfer.py" \
         '--peptide_file' "../preprocess_datasets/preprocessed_datasets/CPTAC_S051_BRCA_onehot_encoded_peptide_quants.tsv" \
         '--n_runs' "35" '--seq_length' "60" \
         '--output_file' "all_TMT11_lumos_datasets/all_TMT11_lumos_datasets" \
         '--filter_size' '7' \
         '--n_filters' '5' \
         '--n_layers' '4' \
         '--n_nodes' '80' \
         '--dropout' '0.25' \
         '--learning_rate' '0.005' \
         '--batch' '500' \
         '--random_run' '0'
            
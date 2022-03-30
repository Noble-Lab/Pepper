
#Train 1-mer model
python3 "1mer_count_linear_coefficient_predictor.py" \
         '--peptide_file' "../preprocess_datasets/preprocessed_datasets/2019_guo_nci60_formatted_peptide_quants.tsv" \
         '--n_runs' "120" '--seq_length' "60" 
         
#Train 2-mer model
python3 "2mer_count_linear_coefficient_predictor.py" \
         '--peptide_file' "../preprocess_datasets/preprocessed_datasets/2019_guo_nci60_formatted_peptide_quants.tsv" \
         '--n_runs' "120" '--seq_length' "60" 
         
#Train 3-mer model
python3 "3mer_count_linear_coefficient_predictor.py" \
         '--peptide_file' "../preprocess_datasets/preprocessed_datasets/2019_guo_nci60_formatted_peptide_quants.tsv" \
         '--n_runs' "120" '--seq_length' "60" 
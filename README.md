# MBZUAI-AI701-GroupProject

1 Homophily
For homophily datasets- Cora, Citeseer and PubMed,
1) Go to Homophily/
2) run command: **python generate_dataset.py --dset_name ${dataset_name}** to generate datasetm choice is ('Cora','CiteSeer','PubMed')
3) run command: **python model.py --layer_num ${layer_num} --dataset_path dataset_name** , e.g., 
*python model.py --layer_num 3 --dataset_path data/Cora_processed/*
it will run 4 trials and output the mean and max accuracy for the dataset.

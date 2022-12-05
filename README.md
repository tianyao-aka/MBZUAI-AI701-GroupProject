# MBZUAI-AI701-GroupProject

### Homophily dataset

For homophily datasets: Cora, Citeseer and PubMed,
1) Go to Homophily/
2) run command: ***python generate_dataset.py --dset_name ${dataset_name}*** to generate dataset. choice is ('Cora','CiteSeer','PubMed')
3) run command: ***python model.py --layer_num ${layer_num} --dataset_path ${dataset_name}*** , e.g., 

   ```python model.py --layer_num 3 --dataset_path data/Cora_processed/```

Each model will run 4 trials and output the mean and max accuracy for the dataset.

### Heterophily dataset

For heterophily datasets: Texas and Wisconsin are in the WebKB benchmark dataset, and Chameleon is in the Wikipedia benchmark dataset.
1) Go to Heterophily/
2) modify the dataset name parameter in ```load_data``` as Texas and Wisconsin in [utils_webkb.py](https://github.com/tianyao-aka/MBZUAI-AI701-GroupProject/blob/main/Heterophily/utils_webkb.py#L302) or Chameleon in  [utils_wiki.py](https://github.com/tianyao-aka/MBZUAI-AI701-GroupProject/blob/main/Heterophily/utils_wiki.py#L301). then run the command ```python utils_webkb.py``` or  ```python utils_wiki.py``` to  preprecess graph data in datasets.
3) modify the dataset name parameter in [model_webkb.py](https://github.com/tianyao-aka/MBZUAI-AI701-GroupProject/blob/main/Heterophily/model_webkb.py) or [model_wiki.py](https://github.com/tianyao-aka/MBZUAI-AI701-GroupProject/blob/main/Heterophily/model_wiki.py), then  run command: 

   ```python model.py --layer_num ${layer_num} --dataset_path ${dataset_name}```

Each model will run 10 trials and output the mean and max accuracy for the dataset.


The processed data of Texas and Wisconsin dataset and the model training log files are uploaded into the [OneDrive](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zeyuan_yin_mbzuai_ac_ae/Er4xO66AB09Dqt2iMh31a4IBI_ha3PLjOMMbHIBqfnQWSg?e=4zLmLt). (People in MBZUAI with the link can view)

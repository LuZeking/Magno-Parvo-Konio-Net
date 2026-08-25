## How to add a new dataset as a metric

1. define .py to load the dataset
2. define info mapping from path name in info_mappings.py
3. import also in __init__.py to register
4. define the experiment condition and plotting in experiments.py
5. if models trained on different dataset, need to define possibility-to-class in experiments.py
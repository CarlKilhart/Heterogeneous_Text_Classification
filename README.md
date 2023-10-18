# Heterogeneous Text Classification
## Environment Requirement
python==3.8.0  
torch==2.0.1  
dgl==1.1.1  
numpy==1.24.3  
pandas==1.5.3  
scipy==1.10.1  
scikit-learn=1.2.2  
tqdm==4.65.0

## How to Run
Create a directory named *model* in the root directory before running the code.

### Examples

Run our model on ACM dataest:  
```bash
python main.py --dataset ACM --model_name hgcn_ACM --device 0
```

### Arguments
Comments of arguments in `main.py`.
- **--dataset**: The name of the dataset. All processed dataset should be put in `data/dataset`.
- **--num_epochs**: The max number of epochs for training, default as 100.
- **--num_trials**: The number of trials to evaluate the stability, default as 10.
- **--device**: Positive integers for corresponding gpu and -1 for cpu.
- **--model_name**: The name of the file to store the model.
- **--load_path**: The path to load a pretrained model if needed.
- **--add_feature**: Load pretrained features from the dataset if `True`.
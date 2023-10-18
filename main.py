from utils.metrics import get_f1
import pickle
import argparse
import torch
from utils.helper import logging_helper
from codes.model import HGCNModel

def train_eval_model(model_class, model_args, train_args, eval_args, num_trials):
    # Get data
    
    logging = logging_helper()
    for eval_arg in eval_args:
        logging.register(f'{eval_arg[0]}_f1', mode='mean')
    for i in range(num_trials):
        print(f'Trial {i}>')
        model = model_class(**model_args)
        
        model.train(**train_args)
        
        predict_dict = {}
        for eval_arg in eval_args:
            with torch.no_grad():
                data_name = eval_arg[0]
                data = eval_arg[1]
                label = eval_arg[2]
                predict = model.eval(**data).cpu()
                score = get_f1(predict, label)
                predict_dict[data_name] = predict
                logging.log(f'{data_name}_f1', score)
        logging.step_output(epoch_name=None, refresh=False)
    logging.final_output(std=True)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='The dataset name for textgcn model')
    parser.add_argument('--load_path', type=str, default=None, help='The path of model to load')
    parser.add_argument('--model_name', type=str, help='The name of the model')
    parser.add_argument('--num_epochs', type=int, default=100, help='The number of epoches for training')
    parser.add_argument('--num_trials', type=int, default=10, help='The number of trials')
    parser.add_argument('--device', type=int, default=0, help='The device for pytorch operations')
    parser.add_argument('--add_feature', type=bool, default=False, help='Whether to add feature in textgcn model')
    args = parser.parse_args()
    
    import time
    start = time.time()
    
    # Load graph and vocab
    with open(f'data/dataset/{args.dataset}/graph.pkl', 'rb') as f:
        dataset = pickle.load(f)
    graph_list = dataset['graph_list']
    buf = dataset['buf']
    feature = dataset['feature'] if args.add_feature else None
    
    # Load data
    data_path = f'data/dataset/{args.dataset}/dataset.pkl'
    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)
    trainset = dataset['trainset']
    testset = dataset['testset']
    
    # Convert data to list
    train_data = trainset['text'].tolist()
    test_data = testset['text'].tolist()
    train_label = trainset['label'].tolist()
    test_label = testset['label'].tolist()
    label_code = trainset['label'].unique()
    num_labels = len(label_code)
    num_words = len(buf.vocab_set)
    
    # Generate index
    train_index = list(range(len(train_data)))
    test_index = list(range(len(train_data), len(train_data)+len(test_data)))
    
    if args.device == -1:
        device = 'cpu'
    else:
        device = f'cuda:{args.device}'
    model = HGCNModel
    model_args = {
        'graph_list': graph_list,
        'target_index': train_index+test_index,
        'buf': buf,
        'feature_list': feature,
        'load_path': args.load_path,
        'num_labels': num_labels,
        'device': device
    }
    train_args = {
        'train_index': train_index,
        'train_label': train_label,
        'model_name': args.model_name,
        'num_epochs': args.num_epochs
    }
    eval_list = []
    eval_args = {
        'index': train_index,
    }
    eval_list.append(('Train', eval_args, train_label))
    eval_args = {
        'index': test_index,
    }
    eval_list.append(('Test', eval_args, test_label))
    
    train_eval_model(model, model_args, train_args, eval_list, args.num_trials)
    
    end = time.time()
    t = end - start
    print(f'Total time: {t:.2f}s')
    print('Done!')
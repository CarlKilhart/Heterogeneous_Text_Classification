import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from sklearn.model_selection import train_test_split
from utils.gcn import HGCN
from utils.helper import logging_helper, earlystop_helper
from utils.metrics import get_accuracy

DIM_INITIAL_FEATURE = 3000      # Dim of initial feature for each node in graph

class HGCNModel:
    def __init__(self, graph_list:list, buf, target_index:list, feature_list:list=None, dim_hidden:int=200, num_labels:int=2, load_path:str=None, device:str='cuda:0') -> None:
        self.device = torch.device(device)
        self.buf = buf
        
        if feature_list is None:
            feature_list = [torch.randn(graph.number_of_nodes(), DIM_INITIAL_FEATURE).to(self.device) for graph in graph_list]
        elif not len(graph_list) == len(feature_list):
            raise ValueError('graph_list and feature_list must have the same length')
        self.graph_list = [graph.to(self.device) for graph in graph_list]
        self.feature_list = feature_list
        
        self.model = HGCN(num_graph=len(graph_list), target_index=target_index, dim_in=self.feature_list[0].shape[1], dim_hidden=dim_hidden, dim_out=num_labels).to(self.device)
        if load_path is not None:
            self.model.load_state_dict(torch.load(load_path))
        
    def train(self, train_index:torch.Tensor, train_label:torch.Tensor, model_name:str, val_split:bool=True, num_epochs:int=100, lr:float=0.01, early_stop:int=10):
        if val_split:
            train_index, val_index, train_label, val_label = train_test_split(train_index, train_label, test_size=0.2, random_state=42)
        
        if not isinstance(train_label, torch.Tensor):
            train_label = torch.tensor(train_label, dtype=torch.long).to(self.device)
        if not isinstance(val_label, torch.Tensor):
            val_label = torch.tensor(val_label, dtype=torch.long).to(self.device)
        
        loss_func = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: 0.9**e)
        
        logging = logging_helper()
        logging.register('loss')
        logging.register('train_acc')
        if val_split:
            logging.register('val_acc')
        earlystopping = earlystop_helper(early_stop, mode='max')

        for _ in range(num_epochs):
            self.model.train()
            
            logits = self.get_logits(train_index)
            loss = loss_func(logits, train_label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_acc = get_accuracy(train_label, logits.argmax(dim=1))
            
            # log
            logging.log('loss', loss.cpu().item())
            logging.log('train_acc', train_acc.cpu().item())
            
            # val
            if val_split:
                self.model.eval()
                logits = self.get_logits(val_index)
                val_acc = get_accuracy(val_label, logits.argmax(dim=1))
                # log
                logging.log('val_acc', val_acc.cpu().item())
                
            if val_split:
                flag = earlystopping.update(val_acc)
                if flag == 1:
                    self.save(f'model/hgcn/{model_name}.pkl')
                elif flag == -1:
                    logging.step_output(refresh=False)
                    break
                
                logging.step_output()
        
        self.load(f'model/hgcn/{model_name}.pkl')
    
    def get_logits(self, index:torch.Tensor):
        logits = self.model(self.graph_list, self.feature_list)
        return logits[index]
    
    def eval(self, index:torch.Tensor):
        logits = self.get_logits(index)
        predict = logits.argmax(dim=1)
        
        return predict
    
    def load(self, load_path:str):
        self.model.load_state_dict(torch.load(load_path))
    
    def save(self, save_path:str):
        torch.save(self.model.state_dict(), save_path)
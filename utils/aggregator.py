import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SelfAttention, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.query = nn.Linear(in_dim, out_dim)
        self.key = nn.Linear(in_dim, out_dim)
        self.value = nn.Linear(in_dim, out_dim)
        self.softmax = nn.Softmax(dim=2)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        query = self.query(x).view(batch_size, seq_len, self.out_dim)
        key = self.key(x).view(batch_size, seq_len, self.out_dim)
        value = self.value(x).view(batch_size, seq_len, self.out_dim)
        attention = torch.matmul(query, key.transpose(1, 2))
        attention = self.softmax(attention)
        context = torch.matmul(attention, value)
        out = self.gamma * context + x
        return out.mean(dim=1)
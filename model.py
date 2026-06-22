import torch
from torch import nn

class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()
        # Define the RNN layer
        self.rnn = nn.RNN(
            input_size=1080,  
            hidden_size=240,  
            num_layers=1,     # Number of recurrent layers
            nonlinearity='relu', 
            bias=True     
        )
        # Define the output layer
        self.output = nn.Linear(
            in_features=240,
            out_features=24   # Number of output features (number of classes)
        )
    def forward(self, x):
        y, hidden = self.rnn(x)
  
        x = self.output(y)
        
        return x

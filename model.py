
import torch.nn as nn


class ForexLSTM(nn.Module):
    def __init__(self, input_size=14, hidden_size=8, num_layers=2):
        super().__init__()
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,      # Number of input features
            hidden_size=hidden_size,     # Size of LSTM hidden state
            num_layers=num_layers,       # Number of LSTM layers
            batch_first=True,            # Batch dimension first
            dropout=0.4                  # Dropout for regularization
        )
        # First linear layer
        self.linear1 = nn.Linear(hidden_size, 4)
        self.relu = nn.ReLU()  # Activation function
        self.dropout = nn.Dropout(0.3)  # Dropout layer
        self.linear2 = nn.Linear(4, 1)  # Output layer
        # self.sigmoid = nn.Sigmoid()
        self.batch_norm = nn.BatchNorm1d(hidden_size)  # Batch normalization removed
    
    def forward(self, x):
        # Process through LSTM
        lstm_out, _ = self.lstm(x)
        # Get last output
        last_output = lstm_out[:, -1, :]
        # Apply batch normalization
        normalized = self.batch_norm(last_output)
        # Dense layers
        x = self.linear1(normalized) #  normalized
        # x = self.linear1(last_output)
        # x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        # x = self.sigmoid(self.linear2(x)) 
        return x.squeeze(-1)
    
    
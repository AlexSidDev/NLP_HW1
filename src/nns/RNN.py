import torch.nn as nn


class MyLSTM(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, num_classes=3, add_data_dim=None):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, 
                            num_layers=num_layers, batch_first=True, 
                            dropout=0.05)
        if add_data_dim is not None:
            self.add_projection = nn.Linear(add_data_dim, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x, add_data=None):
        if add_data is not None:
            add_data = self.add_projection(add_data)
        x, _ = self.lstm(x)
        x = self.classifier(x[:, -1] + (add_data if add_data is not None else 0)) # using last token from sequence
        return x
        
import torch.nn as nn


class MyLSTM(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, 
                            num_layers=num_layers, batch_first=True, 
                            dropout=0.05)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.classifier(x[:, -1]) # using last token from sequence
        return x
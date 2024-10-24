import torch.nn as nn


class MyCNN(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, num_classes=3):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layer = nn.Sequential(
                nn.Conv1d(input_dim, hidden_size, 3, padding=1),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU()
            )
            layers.append(layer)
            input_dim = hidden_size
        self.layers = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.layers(x.transpose(2, 1))
        x = x.mean(-1) # global avg pool
        x = self.classifier(x)
        return x
# load_or_create_model.py
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, fc_size):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_size2, fc_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out = self.fc1(out[:, -1, :])  # Take last time step
        out = self.relu(out)
        out = self.fc2(out)
        return out

def load_model(file_path, device):
    checkpoint = torch.load(file_path, map_location=device, weights_only=False)
    model = LSTMModel(checkpoint['input_size'], 64, 32, 16)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    return model

def load_or_create_model(model_path, device, dataset):
    try:
        model = load_model(model_path, device)
        print("Loaded existing model for retraining")
    except FileNotFoundError:
        model = LSTMModel(dataset.dimension(), 64, 32, 16).to(device)
        print("Initialized new model")

    return model
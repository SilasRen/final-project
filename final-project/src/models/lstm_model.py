import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

class LSTMReg(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def train_lstm(X, y, hidden_size=64, num_layers=1, lr=1e-3, epochs=5, batch_size=64, device='cpu', verbose=True):
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
    ds = torch.utils.data.TensorDataset(X, y)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = LSTMReg(1, hidden_size, num_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    
    if verbose:
        pbar = tqdm(range(epochs), desc='Training LSTM')
    else:
        pbar = range(epochs)
    
    for epoch in pbar:
        epoch_loss = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        if verbose:
            pbar.set_postfix({'loss': f'{epoch_loss/len(dl):.4f}'})
    
    model.eval()
    return model

def save_lstm(model, path):
    """Save LSTM model"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f'[SAVE] LSTM model saved to {path}')

def load_lstm(path, hidden_size=64, num_layers=1, device='cpu'):
    """Load LSTM model"""
    model = LSTMReg(1, hidden_size, num_layers).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f'[LOAD] LSTM model loaded from {path}')
    return model

def predict_lstm(model, X, device='cpu'):
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(device)
    with torch.no_grad():
        pred = model(X).cpu().numpy().ravel()
    return pred

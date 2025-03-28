# Waah kya dimag lagaya hai
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class StockDataset(Dataset):
    def __init__(self, X, y, window_size=10):
        self.X = X
        self.y = y
        self.window_size = window_size
        
    def __len__(self):
        return len(self.X) - self.window_size
    
    def __getitem__(self, idx):
        features = self.X[idx:idx+self.window_size]
        target = self.y[idx+self.window_size]
        return torch.FloatTensor(features), torch.FloatTensor(target)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out

def create_features(data, window_size=5):
    df = data.copy()
    df['Diff_1'] = df['Close'].diff()
    df['Diff_2'] = df['Diff_1'].diff()
    df['MA_5'] = df['Close'].rolling(window=window_size).mean()
    df['MA_10'] = df['Close'].rolling(window=window_size*2).mean()
    df['Volatility'] = df['Close'].rolling(window=window_size).std()
    df['Volume_Change'] = df['Volume'].pct_change()
    df.dropna(inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

def scale_dataset(data, feature_scaler, target_scaler, feature_cols):
    X = data[feature_cols].replace([np.inf, -np.inf], np.nan)
    y = data[['Close']].replace([np.inf, -np.inf], np.nan)
    valid_idx = X.notna().all(axis=1) & y.notna().all(axis=1)
    X = X[valid_idx]
    y = y[valid_idx]
    X_scaled = feature_scaler.transform(X)
    y_scaled = target_scaler.transform(y)
    return X_scaled, y_scaled

def train_model(model, train_loader, val_loader, num_epochs, patience, device):
    best_val_loss = float('inf')
    best_model = None
    epochs_no_improve = 0
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item()
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
    
    model.load_state_dict(best_model)
    return model

def evaluate_model(model, test_loader, target_scaler, device):
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(batch_y.cpu().numpy())
    
    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()
    predictions = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    actuals = target_scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()
    
    mse = mean_squared_error(actuals, predictions)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    ssr = np.sum((actuals - predictions) ** 2)
    sst = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = 1 - (ssr / sst) if sst != 0 else 0.0

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2": r2
    }

def predict_next_timestep(model, data, window_size, feature_scaler, target_scaler, feature_cols, device):
    last_window = data.tail(window_size)
    X = feature_scaler.transform(last_window[feature_cols])
    X = torch.FloatTensor(X).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        prediction = model(X)
    return target_scaler.inverse_transform(prediction.cpu().numpy())[0][0]

def big_model_baba(filepath: str):
    # Load and prepare data
    df = pd.read_csv(filepath, parse_dates=['Date'], index_col='Date').sort_index()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # Create features
    featured_df = create_features(df)
    
    # Split data
    n = len(featured_df)
    train_end = int(n * 0.7)
    val_end = train_end + int(n * 0.15)
    train_data = featured_df.iloc[:train_end]
    val_data = featured_df.iloc[train_end:val_end]
    test_data = featured_df.iloc[val_end:]
    
    # Scale data
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    feature_cols = ['Open', 'High', 'Low', 'Volume', 'Diff_1', 'Diff_2', 'MA_5', 'MA_10', 'Volatility', 'Volume_Change']
    
    target_scaler.fit(train_data[['Close']])
    feature_scaler.fit(train_data[feature_cols])
    
    X_train, y_train = scale_dataset(train_data, feature_scaler, target_scaler, feature_cols)
    X_val, y_val = scale_dataset(val_data, feature_scaler, target_scaler, feature_cols)
    X_test, y_test = scale_dataset(test_data, feature_scaler, target_scaler, feature_cols)
    
    # Create datasets
    window_size = 10
    batch_size = 16
    train_dataset = StockDataset(X_train, y_train, window_size)
    val_dataset = StockDataset(X_val, y_val, window_size)
    test_dataset = StockDataset(X_test, y_test, window_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize and train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = X_train.shape[1]
    model = LSTMModel(input_size, 64, 2, 1, 0.2).to(device)
    trained_model = train_model(model, train_loader, val_loader, 200, 20, device)
    
    # Evaluate model
    stats = evaluate_model(trained_model, test_loader, target_scaler, device)
    
    # Train on full dataset for next timestep prediction
    full_X, full_y = scale_dataset(featured_df, feature_scaler, target_scaler, feature_cols)
    full_dataset = StockDataset(full_X, full_y, window_size)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
    full_model = LSTMModel(input_size, 64, 2, 1, 0.2).to(device)
    full_optimizer = torch.optim.Adam(full_model.parameters(), lr=0.001)
    full_criterion = nn.MSELoss()
    
    for epoch in range(100):
        full_model.train()
        for batch_X, batch_y in full_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = full_model(batch_X)
            loss = full_criterion(outputs, batch_y)
            full_optimizer.zero_grad()
            loss.backward()
            full_optimizer.step()
    
    # Predict next timestep
    next_price = predict_next_timestep(full_model, featured_df, window_size, 
                                     feature_scaler, target_scaler, feature_cols, device)
    
    return next_price, stats
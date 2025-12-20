import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Data Preparation
# ==========================================

def generate_synthetic_data(length=1000):
    """Generates a sine wave with some noise."""
    t = np.linspace(0, 100, length)
    data = np.sin(t) + np.random.normal(0, 0.05, length)
    return data.astype(np.float32)

def create_sliding_windows(data, window_size):
    """
    Creates sliding window pairs (X, y).
    X: sequence of `window_size` steps
    y: the next value (step window_size + 1)
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    
    return np.array(X), np.array(y)

# hyperparameters
WINDOW_SIZE = 20
TEST_SPLIT = 0.2
BATCH_SIZE = 32
LEARNING_RATE = 0.01
EPOCHS = 50

# Generate data
raw_data = generate_synthetic_data()
X, y = create_sliding_windows(raw_data, WINDOW_SIZE)

# Chronological Split (Train/Test) - NO SHUFFLING
split_index = int(len(X) * (1 - TEST_SPLIT))

X_train_np, X_test_np = X[:split_index], X[split_index:]
y_train_np, y_test_np = y[:split_index], y[split_index:]

# Convert to PyTorch Tensors
# LSTM expects input shape: (batch_size, sequence_length, input_size)
# Here input_size is 1 (univariate time series)
X_train = torch.tensor(X_train_np).unsqueeze(-1) # Shape: (N_train, WINDOW_SIZE, 1)
y_train = torch.tensor(y_train_np).unsqueeze(-1) # Shape: (N_train, 1)

X_test = torch.tensor(X_test_np).unsqueeze(-1)
y_test = torch.tensor(y_test_np).unsqueeze(-1)

print(f"Train shapes - X: {X_train.shape}, y: {y_train.shape}")
print(f"Test shapes  - X: {X_test.shape}, y: {y_test.shape}")

# ==========================================
# 2. Model Definition
# ==========================================

class TimeSeriesLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super(TimeSeriesLSTM, self).__init__()
        self.hidden_size = hidden_size
        # batch_first=True means input shape is (batch, seq, feature)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch, seq, feature)
        # out shape: (batch, seq, hidden_size)
        # _ (hn, cn) are hidden and cell states (not used here)
        out, _ = self.lstm(x)
        
        # We only care about the output of the lat time step to predict the next value
        last_time_step_out = out[:, -1, :] 
        prediction = self.fc(last_time_step_out)
        return prediction

model = TimeSeriesLSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==========================================
# 3. Training Loop
# ==========================================

print("\nStarting training...")
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
# Note: Shuffle=False makes sense for time-series if stateful, but for sliding window 
# where each sample is independent (stateless LSTM use), shuffling training batches is actually fine and often better.
# However, user requested "chronological (no shuffling) train/test split".
# Usually for training *within* the train set, shuffling is okay for independent windows, 
# but strictly speaking strict chronological processing would mean shuffle=False.
# I'll stick to shuffle=False to be safe with "chronological" instruction.

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch+1) % 10 == 0:
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.6f}")

# ==========================================
# 4. Evaluation
# ==========================================

print("\nEvaluating...")
model.eval()
with torch.no_grad():
    test_predictions = model(X_test)
    test_loss = criterion(test_predictions, y_test)

print(f"Test MSE: {test_loss.item():.6f}")

# Optional: Visualize
# Plotting the first 100 test points
plt.figure(figsize=(10, 6))
plt.plot(y_test_np, label='Actual Data')
plt.plot(test_predictions.numpy(), label='Predictions')
plt.title("Time Series Prediction (Next Step)")
plt.legend()
plt.savefig("prediction_plot.png")
print("Plot saved to prediction_plot.png")

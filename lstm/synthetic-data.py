import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Create synthetic data
np.random.seed(42)
torch.manual_seed(42)

# Generate some random data
data = np.random.randn(100, 1, 1).astype(np.float32)
labels = np.sin(np.arange(100) * 0.1).reshape(-1, 1).astype(np.float32)

# Convert data to PyTorch tensors
data_tensor = torch.from_numpy(data)
label_tensor = torch.from_numpy(labels)

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Set hyperparameters
input_size = 1
hidden_size = 100
output_size = 1
learning_rate = 0.01
num_epochs = 100

# Initialize the model, loss function, and optimizer
model = LSTMModel(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(data_tensor)
    loss = criterion(outputs, label_tensor)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Test the model
with torch.no_grad():
    test_data = torch.from_numpy(np.random.randn(10, 1, 1).astype(np.float32))
    predicted = model(test_data)

print("Predicted:")
print(predicted)

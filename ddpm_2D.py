import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# Function to load and preprocess data
def load_and_preprocess_data(file_path, batch_size=32):
    # Load the data
    data = pd.read_csv(file_path, header=None)

    # Split the space-separated string into individual numbers
    data = data[0].str.split(expand=True).astype(float)

    # Convert to PyTorch tensors
    data_tensor = torch.tensor(data.values, dtype=torch.float32)

    # Ensure that each row has 200 values
    assert data_tensor.shape[1] == 200, "Each data row must have 200 values."

    # Create a TensorDataset
    dataset = TensorDataset(data_tensor)

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


# Function to add noise to the data
def q_sample(x_start, t, noise, beta_t):
    # Reshape beta_t[t] to match the dimensions of x_start and noise
    beta_t_t = beta_t[t].view(-1, 1).to(x_start.device)
    return torch.sqrt(1 - beta_t_t) * x_start + torch.sqrt(beta_t_t) * noise


# Set the total number of time steps for the diffusion process
T = 1000

# Use PyTorch to create the beta_t tensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
beta_t = torch.linspace(0.0001, 0.02, T, device=device)


# Training function
def train_ddpm(model, dataloader, num_epochs=100):
    # Use Mean Squared Error (MSE) as the loss function
    criterion = nn.MSELoss()
    # Use Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # List to store loss for visualization
    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in dataloader:
            x_start = batch[0].to(device)  # Ensure x_start is on the correct device
            # Randomly select a time step t
            t = torch.randint(0, T, (x_start.size(0),), device=x_start.device).long()
            # Generate Gaussian noise with the same shape as x_start
            noise = torch.randn_like(x_start)
            # Add noise to x_start at time step t to get x_noisy
            x_noisy = q_sample(x_start, t, noise, beta_t)
            optimizer.zero_grad()
            # Use the model to predict the denoised data
            x_recon = model(x_noisy)
            # Calculate the loss (MSE between predicted and original data)
            loss = criterion(x_recon, x_start)
            epoch_loss += loss.item()
            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

        # Average loss for the epoch
        avg_epoch_loss = epoch_loss / len(dataloader)
        losses.append(avg_epoch_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_epoch_loss}")

    # Plot the loss over epochs
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.show()

    # Save the model weights
    torch.save(model.state_dict(), 'ddpm_model_weights.pth')
    print("Model weights saved to 'ddpm_model_weights.pth'")


# Define the model architecture
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # Define the layers of the MLP
        self.model = nn.Sequential(
            nn.Linear(200, 256),  # Input layer, maps 200 dimensions to 256
            nn.ReLU(),  # Activation function
            nn.Linear(256, 256),  # Hidden layer, 256 dimensions to 256
            nn.ReLU(),  # Activation function
            nn.Linear(256, 200)  # Output layer, maps 256 dimensions back to 200
        )

    def forward(self, x):
        return self.model(x)  # Forward pass


# Example usage
file_path = './dataset/pedestrians_positions_MI.csv'
dataloader = load_and_preprocess_data(file_path, batch_size=32)

# Instantiate the model and move it to the device (GPU/CPU)
model = MLP().to(device)

# Train the model
train_ddpm(model, dataloader)

# Save the model weights after training
torch.save(model.state_dict(), 'ddpm_model_weights.pth')
print("Model weights saved to 'ddpm_model_weights.pth'")

# OR save the entire model
torch.save(model, 'ddpm_model.pth')
print("Entire model saved to 'ddpm_model.pth'")

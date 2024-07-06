import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
# Function to create sinusoidal time embeddings
def get_sinusoidal_embeddings(timesteps, embedding_dim):
    half_dim = embedding_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return emb
def get_sinusoidal_embeddings(timesteps, embedding_dim):
    half_dim = embedding_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return emb

# Define the model architecture with time embedding
class MLP(nn.Module):
    def __init__(self, time_emb_dim=256):
        super(MLP, self).__init__()
        self.time_emb_dim = time_emb_dim
        self.time_embedding = nn.Linear(time_emb_dim, time_emb_dim)
        self.model = nn.Sequential(
            nn.Linear(200 + time_emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 200)
        )

    def forward(self, x, t):
        t_emb = get_sinusoidal_embeddings(t, self.time_emb_dim).to(x.device)
        t_emb = self.time_embedding(t_emb)
        x = torch.cat([x, t_emb], dim=1)
        return self.model(x)

# Function to load the model and weights
def load_and_preprocess_data(file_path, batch_size=32):
    data = pd.read_csv(file_path, header=None)
    data = data[0].str.split(expand=True).astype(float)

    # Normalize the data to [0, 1] range
    data_min = data.min().min()
    data_max = data.max().max()
    data = (data - data_min) / (data_max - data_min)

    data_tensor = torch.tensor(data.values, dtype=torch.float32)
    assert data_tensor.shape[1] == 200, "Each data row must have 200 values."
    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, data_min, data_max

# Function to generate new samples using the model
def generate_samples(model, num_samples, T, beta_t, device):
    model.eval()
    with torch.no_grad():
        # Start from Gaussian noise
        x = torch.randn(num_samples, 200, device=device)
        for t in reversed(range(T)):
            t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
            x = model(x, t_tensor)
        return x

# Function to visualize generated samples
def visualize_samples(samples):
    for sample in samples:
        x_coords = sample[::2].cpu()
        y_coords = sample[1::2].cpu()
        plt.scatter(x_coords, y_coords)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Generated Pedestrian Positions')
    plt.show()

# Function to save generated samples to a CSV file
def save_samples_to_csv(samples, output_file):
    samples = samples.cpu().numpy()  # Convert samples to NumPy array
    df = pd.DataFrame(samples)
    df.to_csv(output_file, sep=' ', index=False, header=False)
    print(f"Generated samples saved to {output_file}")

# Training function with time embedding
def train_ddpm(model, dataloader, num_epochs=200):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in dataloader:
            x_start = batch[0].to(device)
            t = torch.randint(0, T, (x_start.size(0),), device=x_start.device).long()
            noise = torch.randn_like(x_start, device=device)
            x_noisy = q_sample(x_start, t, noise, beta_t)
            optimizer.zero_grad()
            x_recon = model(x_noisy, t)
            loss = criterion(x_recon, x_start)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_epoch_loss = epoch_loss / len(dataloader)
        losses.append(avg_epoch_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_epoch_loss}")

    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.show()
    torch.save(model.state_dict(), 'ddpm_model_weights.pth')
    print("Model weights saved to 'ddpm_model_weights.pth'")

# Function to add noise to the data
def q_sample(x_start, t, noise, beta_t):
    beta_t_t = beta_t[t].view(-1, 1).to(x_start.device)
    return torch.sqrt(1 - beta_t_t) * x_start + torch.sqrt(beta_t_t) * noise

# Function to load and preprocess data
def load_and_preprocess_data(file_path, batch_size=32):
    data = pd.read_csv(file_path, header=None)
    data = data[0].str.split(expand=True).astype(float)
    data_tensor = torch.tensor(data.values, dtype=torch.float32)
    assert data_tensor.shape[1] == 200, "Each data row must have 200 values."
    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
if __name__ == "__main__":
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = 1000
    beta_t = torch.linspace(0.0001, 0.02, T, device=device)

    # Load the dataset
    file_path = './dataset/pedestrians_positions_MI.csv'
    dataloader = load_and_preprocess_data(file_path, batch_size=32)

    # Instantiate the model and move it to the device
    model = MLP().to(device)

    # Train the model
    train_ddpm(model, dataloader)

    # Generate new samples
    num_samples = 10  # Number of samples to generate
    generated_samples = generate_samples(model, num_samples, T, beta_t, device)

    # Visualize the generated samples
    visualize_samples(generated_samples)

    # Save the generated samples to a CSV file
    output_file = 'generated_pedestrians_positions.csv'
    save_samples_to_csv(generated_samples, output_file)

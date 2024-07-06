import torch
from utils import visualize_samples, save_samples_to_csv, get_sinusoidal_embeddings
from models.unet import UNet1D
from utils import load_and_preprocess_data

# Function to generate new samples using the model
def generate_samples(model, num_samples, T, beta_t, device):
    model.eval()
    with torch.no_grad():
        # Start from Gaussian noise
        x = torch.randn(num_samples, 1, 10, 20, device=device)
        for t in reversed(range(T)):
            t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
            x = model(x, t_tensor)
        return x

# Example usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = 1000
    beta_t = torch.linspace(0.0001, 0.02, T, device=device)

    # Load the trained model
    model = UNet1D(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load('ddpm_unet_weights.pth'))
    model.eval()  # Set the model to evaluation mode

    # Generate new samples
    num_samples = 10  # Number of samples to generate
    generated_samples = generate_samples(model, num_samples, T, beta_t, device)

    # Visualize the generated samples
    visualize_samples(generated_samples)

    # Save the generated samples to a CSV file
    output_file = 'generated_pedestrians_positions.csv'
    # Load min and max values used during training
    _, data_min, data_max = load_and_preprocess_data('./dataset/pedestrians_positions_MI.csv', batch_size=32)
    save_samples_to_csv(generated_samples, output_file, data_min, data_max)

import torch
from utils import visualize_samples, save_samples_to_csv, get_sinusoidal_embeddings, load_and_preprocess_data
from models.unet import UNet1D


# Function to generate new samples using the model
def generate_samples(model, num_samples, T, device, data_min, data_max):
    model.eval()
    with torch.no_grad():
        # Start from Gaussian noise
        x = torch.randn(num_samples, 2, 100, device=device)  # Ensure the shape matches the training data
        for t in reversed(range(T)):
            t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
            t_emb = get_sinusoidal_embeddings(t_tensor, model.time_emb_dim, device)
            x = model(x, t_emb)

        # Rescale the samples back to original range
        x = x * (data_max - data_min) + data_min
        return x


# Example usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = 1000

    # Load the trained model
    model = UNet1D(in_channels=2, out_channels=2).to(device)  # Ensure the model's input and output channels match
    model.load_state_dict(torch.load('ddpm_unet1d_weights_final.pth'))
    model.eval()  # Set the model to evaluation mode

    # Load min and max values used during training
    dataloader, data_min, data_max = load_and_preprocess_data('./dataset/pedestrians_positions_MI_converted.csv',
                                                              batch_size=32)

    # Generate new samples
    num_samples = 10  # Number of samples to generate
    generated_samples = generate_samples(model, num_samples, T, device, data_min, data_max)

    # Visualize the generated samples
    visualize_samples(generated_samples)

    # Save the generated samples to a CSV file
    output_file = 'generated_pedestrians_positions.csv'
    save_samples_to_csv(generated_samples, output_file, data_min, data_max)

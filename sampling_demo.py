import torch
import matplotlib.pyplot as plt
import pandas as pd
from demo import MLP

# Function to load the model and weights
def load_model(weights_path, device):
    model = MLP().to(device)
    model.load_state_dict(torch.load(weights_path))
    model.eval()  # Set the model to evaluation mode
    return model

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

if __name__ == '__main__':
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = 1000
    beta_t = torch.linspace(0.0001, 0.02, T, device=device)

    # Load the trained model
    model = load_model('ddpm_model_weights.pth', device)

    # Generate new samples
    num_samples = 10  # Number of samples to generate
    generated_samples = generate_samples(model, num_samples, T, beta_t, device)

    # Visualize the generated samples
    visualize_samples(generated_samples)

    # Save the generated samples to a CSV file
    output_file = 'generated_pedestrians_positions.csv'
    save_samples_to_csv(generated_samples, output_file)

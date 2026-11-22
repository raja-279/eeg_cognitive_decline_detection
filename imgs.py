import numpy as np
import matplotlib.pyplot as plt
import os

# 👉 CHANGE THIS to your actual path
file_path = r"C:\Users\Admin\Downloads\eeg_cnn\eeg_cache\sub-073_recurrence_64_imgs.npy"

# Check if file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

# Load dataset
imgs = np.load(file_path)

# Check shape
print("Dataset shape:", imgs.shape)

# Plot first 3 samples safely
num_samples = min(3, len(imgs))

for i in range(num_samples):
    plt.figure(figsize=(3, 3))

    # Convert (C,H,W) → (H,W,C)
    img = imgs[i].transpose(1, 2, 0)

    # Normalize for better visibility
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    plt.imshow(img)
    plt.title(f"Sample {i+1}")
    plt.axis('off')

    # Save image
    plt.savefig(f"sample_{i+1}.png", bbox_inches='tight')
    plt.close()

print("Sample images saved successfully.")
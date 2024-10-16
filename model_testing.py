import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from zipfile import ZipFile
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import segmentation_models_pytorch as smp
from brain_dataset import BrainDataset

# device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# Define transforms for the dataset
def clamp_tensor(x):
    return x.clamp(0, 1)

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]),
    transforms.Lambda(clamp_tensor)
])

# Initialize FastAPI app
app = FastAPI()

# Load datasets
root_dir = '/'
batch_size = 32
workers = 4

test_images_path = 'images/test'
test_masks_path = 'masks/test'
test_dataset = BrainDataset(root_dir, test_images_path, test_masks_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

val_images_path = 'images/valid'
val_masks_path = 'masks/valid'
val_dataset = BrainDataset(root_dir, val_images_path, val_masks_path, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

# Load model
model = smp.Unet(encoder_name="resnet50", encoder_weights='imagenet', in_channels=1, classes=1).to(device)
model_path = "models/best_model.pth"
model.load_state_dict(torch.load(model_path, map_location=device))

# Ensure the directory exists for saving outputs
def create_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Saving images to {output_dir}")

def visualize_input_output_target(input_image, output_image, target_image, output_dir, img_count):
    input_image = input_image.cpu()
    output_image = output_image.cpu()
    target_image = target_image.cpu()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Input Image
    axes[0].imshow(input_image.squeeze().numpy(), cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    # Predicted Output
    axes[1].imshow(output_image.squeeze().numpy(), cmap='gray')
    axes[1].set_title('Output Image (Predicted)')
    axes[1].axis('off')

    # Target Image (Ground Truth)
    axes[2].imshow(target_image.squeeze().numpy(), cmap='gray')
    axes[2].set_title('Target Image (Ground Truth)')
    axes[2].axis('off')

    # Save each figure to the specified output directory with a unique name
    output_path = os.path.join(output_dir, f'output_{img_count}.png')
    fig.savefig(output_path)
    plt.close(fig)

    print(f"Saved visualization {img_count} to {output_path}")

# Function to zip the output images
def zip_output_images(output_dir, zip_filename):
    with ZipFile(zip_filename, 'w') as zipf:
        for foldername, _, filenames in os.walk(output_dir):
            for filename in filenames:
                file_path = os.path.join(foldername, filename)
                zipf.write(file_path, os.path.relpath(file_path, output_dir))
    print(f"Zipped images into {zip_filename}")

@app.get("/process-images/")
async def process_images():
    output_dir = 'output_images'  # Directory where images will be saved
    create_output_dir(output_dir)
    
    model.eval()
    img_count = 0  # Initialize image counter

    with torch.inference_mode():
        for batch, (X, y) in enumerate(val_loader):
            X = X.to(device)
            y = y.to(device)

            y_pred_logits = model(X)
            y_pred_binary = (y_pred_logits > 0.5).float()

            # Increment the image counter for each visualization
            img_count += 1

            # Visualize and save the result in the specified directory
            visualize_input_output_target(X[0], y_pred_binary[0], y[0], output_dir, img_count)
    
    zip_filename = 'output_images.zip'
    zip_filename_path = os.path.join(output_dir, zip_filename)
    zip_output_images(output_dir, zip_filename_path)

    # return FileResponse(zip_filename_path, media_type='application/zip', filename=zip_filename)

# if __name__ == '__main__':
#     # Optionally run FastAPI with uvicorn in development
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

import dagshub
import mlflow

from brain_dataset import BrainDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os
import segmentation_models_pytorch as smp
import torch


def start_or_get_run():
    if mlflow.active_run() is None:
        mlflow.start_run()
    else:
        print(f"Active run with UUID {mlflow.active_run().info.run_id} already exists")


def end_active_run():
    if mlflow.active_run() is not None:
        mlflow.end_run()


def clamp_tensor(x):
    return x.clamp(0, 1)


def main():
    # Retrieve environment variables
    mlflow_username = os.getenv('MLFLOW_TRACKING_USERNAME')
    mlflow_password = os.getenv('MLFLOW_TRACKING_PASSWORD')
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI')

    # Set the environment variables for MLflow
    os.environ['MLFLOW_TRACKING_USERNAME'] = mlflow_username
    os.environ['MLFLOW_TRACKING_PASSWORD'] = mlflow_password
    os.environ['MLFLOW_TRACKING_URI'] = mlflow_uri

    dagshub.init(repo_owner='ignatiusboadi', repo_name='dagshub_proj_II', mlflow=True)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("proj-II-model-training")
    end_active_run()
    start_or_get_run()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    root_dir = '/'
    lr = 0.001
    num_epochs = 10
    batch_size = 32
    workers = 4

    mlflow.log_params({'device': device, 'l_rate': lr, 'epochs': num_epochs, 'batch': batch_size, })

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]),
        transforms.Lambda(clamp_tensor)
    ])

    train_images_path = 'images/train'
    test_images_path = 'images/test'
    val_images_path = 'images/valid'

    train_masks_path = 'masks/train'
    test_masks_path = 'masks/test'
    val_masks_path = 'masks/valid'

    train_dataset = BrainDataset(root_dir, train_images_path, train_masks_path, transform=transform)
    val_dataset = BrainDataset(root_dir, val_images_path, val_masks_path, transform=transform)
    test_dataset = BrainDataset(root_dir, test_images_path, test_masks_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    model = smp.Unet(encoder_name="resnet50", encoder_weights='imagenet', in_channels=1, classes=1).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def to_device(batch, device_type):
        return batch[0].to(device_type), batch[1].to(device_type)

    print('Starting training...')
    best_val_loss = 1
    training_loss = 5
    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0
        for i, batch in tqdm(enumerate(train_loader), total=len(train_loader), leave=True, dynamic_ncols=True):
            input_img, mask = to_device(batch, device)
            output = model(input_img)
            loss = criterion(output, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 5 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}] Average Training Loss: {avg_train_loss:.4f}')

        model.eval()
        valid_loss = 0.0

        with torch.no_grad():
            for inputs, masks in val_loader:
                inputs = inputs.to(device)
                masks = masks.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, masks)
                valid_loss += loss.item()

        avg_valid_loss = valid_loss / len(val_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}] Average Validation Loss: {avg_valid_loss:.4f}')

        if avg_valid_loss < best_val_loss:
            best_val_loss = avg_valid_loss
            torch.save(model.state_dict(), f'models/best_model.pth')
            training_loss = avg_train_loss
    mlflow.log_metrics({'best validation loss': best_val_loss, 'final training loss': training_loss})


if __name__ == '__main__':
    main()

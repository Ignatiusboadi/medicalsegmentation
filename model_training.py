from brain_dataset import BrainDataset
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import segmentation_models_pytorch as smp
import torch


def clamp_tensor(x):
    return x.clamp(0, 1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root_dir = 'processed_data'
    lr = 0.003
    num_epochs = 10
    batch_size = 32
    workers = 4

    image_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]),
        transforms.Lambda(clamp_tensor)
    ])

    mask_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ])

    train_path = os.path.join(root_dir, 'train')
    test_path = os.path.join(root_dir, 'test')
    val_path = os.path.join(root_dir, 'valid')

    train_dataset = BrainDataset(train_path, image_transform=image_transform, mask_transform=mask_transform)
    test_dataset = BrainDataset(test_path, image_transform=image_transform, mask_transform=mask_transform)
    val_dataset = BrainDataset(val_path, image_transform=image_transform, mask_transform=mask_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

    model = smp.Unet(encoder_name="resnet50", encoder_weights='imagenet', in_channels=1, classes=1).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0

        for i, (input_img, mask) in enumerate(train_loader):
            input_img = input_img.to(device)
            mask = mask.to(device)
            output = model(input_img)
            loss = criterion(output, mask)
            optimizer.zero_grad()
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

        # if (epoch + 1) % 5 == 0:
        #     torch.save(model.state_dict(), f'checkpoint_epoch_{epoch + 1}.pth')


if __name__ == '__main__':
    main()

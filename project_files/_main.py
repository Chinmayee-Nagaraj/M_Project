import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm

from _tfdense_unet import TFDenseUNet
from _dataloading import VCTK_DEMAND_Dataset

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainset_dir = "../mp-dataset"
    train_ds = VCTK_DEMAND_Dataset(trainset_dir)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)

    model = TFDenseUNet().to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))

    model.train()  # set model to training mode

    epoch_loss = 0.0

    # -------------------------------
    # 1 Epoch Training
    # -------------------------------
    for clean, noisy, length in tqdm(train_loader, desc="Training", unit="batch"):
        # move to device
        clean = clean.to(device)          # (B, 1, T)
        noisy = noisy.to(device)          # (B, 1, T)

        # remove channel dimension
        clean = clean.squeeze(1)          # (B, T)
        noisy = noisy.squeeze(1)          # (B, T)

        # forward pass
        out_wave = model(noisy)           # (B, T)

        # compute loss
        loss = loss_fn(out_wave, clean)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Average Loss for Epoch: {epoch_loss / len(train_loader):.6f}")

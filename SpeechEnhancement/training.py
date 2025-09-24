import os
import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm

from u_net import UNetWithTFT
from _dataloading import VCTK_DEMAND_Dataset
from stft_utils import STFTProcessor


# -----------------------------
# Training script
# -----------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset + DataLoader
    trainset_dir = "../../dataset/VCTK_DEMAND/trainset"
    train_ds = VCTK_DEMAND_Dataset(trainset_dir)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, 
                              num_workers=2, pin_memory=True)

    # Model, loss, optimizer
    model = UNetWithTFT().to(device)
    stft_processor = STFTProcessor()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))

    # Make checkpoint directory
    os.makedirs("checkpoint", exist_ok=True)

    # Training loop
    num_epochs = 20
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for clean_wave, noisy_wave, _ in pbar:
            noisy_wave = noisy_wave.to(device)  # [B, 1, T]
            clean_wave = clean_wave.to(device)  # [B, 1, T]

            # --- Forward pass ---
            clean_mag, noisy_mag, noisy_phase = stft_processor.stft(clean_wave, noisy_wave)

            # Predict enhanced magnitude from noisy magnitude
            pred_mag = model(noisy_mag.unsqueeze(1).permute(0,1,3,2)) 
            pred_mag = pred_mag.squeeze(1)            # back to [B, F, T]

            # --- Loss in spectrogram domain ---
            loss_mag = loss_fn(pred_mag, clean_mag)

            # --- Reconstruct waveform ---
            rec_wave = stft_processor.istft(pred_mag, noisy_phase, length=clean_wave.shape[-1])

            # --- Loss in waveform domain ---
            loss_wave = loss_fn(rec_wave, clean_wave)

            # Total loss
            loss = loss_mag + loss_wave

            # --- Backprop ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"batch_loss": loss.item()})

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch} - Avg Loss: {avg_loss:.6f}")

        # Save checkpoint every epoch
        checkpoint_path = f"checkpoint/epoch_{epoch}.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
        }, checkpoint_path)
        print(f" Saved checkpoint: {checkpoint_path}")

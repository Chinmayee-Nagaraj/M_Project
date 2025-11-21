import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from torchmetrics.audio.sdr import ScaleInvariantSignalDistortionRatio
from _tfdense_unet import TFDenseUNet
from _dataloading import VCTK_DEMAND_Dataset
import warnings
import matplotlib.pyplot as plt

# Suppress torchaudio warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio._backend.utils")


# ------------------- SI-SDR Loss (non-negative) -------------------
def si_sdr_loss(est, ref, eps=1e-8):
    ref_energy = torch.sum(ref ** 2, dim=1, keepdim=True)
    scale = torch.sum(ref * est, dim=1, keepdim=True) / (ref_energy + eps)
    est_scaled = scale * ref
    noise = est - est_scaled
    si_sdr = 10 * torch.log10(torch.sum(est_scaled**2, dim=1) / (torch.sum(noise**2, dim=1) + eps))
    # Flip and shift to make it non-negative
    return torch.mean(torch.clamp(20 - si_sdr, min=0))


# ------------------- Multi-Resolution STFT Loss -------------------
def multi_res_stft_loss(est, ref):
    """Computes average STFT magnitude loss over multiple FFT sizes."""
    fft_sizes = [256, 512, 1024]
    total_loss = 0.0
    for n_fft in fft_sizes:
        est_spec = torch.stft(est, n_fft=n_fft, hop_length=n_fft // 4, return_complex=True)
        ref_spec = torch.stft(ref, n_fft=n_fft, hop_length=n_fft // 4, return_complex=True)
        total_loss += torch.mean((torch.abs(est_spec) - torch.abs(ref_spec)) ** 2)
    return total_loss / len(fft_sizes)


# ------------------- Evaluation -------------------
def evaluate(model, val_loader, device, loss_fn, pesq_metric, stoi_metric, sisdr_metric):
    model.eval()
    val_loss = 0.0
    pesq_scores, stoi_scores, sisdr_scores = [], [], []

    with torch.no_grad():
        for clean, noisy, length in val_loader:
            clean = clean.to(device).squeeze(1)
            noisy = noisy.to(device).squeeze(1)
            out_wave = model(noisy)

            # Combined loss
            loss = (
                0.6 * loss_fn(out_wave, clean)
                + 0.2 * si_sdr_loss(out_wave, clean)
                + 0.2 * multi_res_stft_loss(out_wave, clean)
            )
            val_loss += loss.item()

            est, ref = out_wave.cpu(), clean.cpu()
            try:
                pesq_scores.append(pesq_metric(ref, est).item())
                stoi_scores.append(stoi_metric(ref, est).item())
                sisdr_scores.append(sisdr_metric(ref, est).item())
            except Exception:
                pass

    avg_loss = val_loss / len(val_loader)
    avg_pesq = sum(pesq_scores) / len(pesq_scores) if pesq_scores else 0.0
    avg_stoi = sum(stoi_scores) / len(stoi_scores) if stoi_scores else 0.0
    avg_sisdr = sum(sisdr_scores) / len(sisdr_scores) if sisdr_scores else 0.0

    # Reset metrics
    pesq_metric.reset()
    stoi_metric.reset()
    sisdr_metric.reset()

    return avg_loss, avg_pesq, avg_stoi, avg_sisdr


# ------------------- Training -------------------
def train_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Dataset
    dataset_dir = "../mp-dataset"
    full_ds = VCTK_DEMAND_Dataset(dataset_dir)
    val_size = int(0.15 * len(full_ds))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

    # Model, optimizer, scheduler
    model = TFDenseUNet().to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # Metrics
    pesq_metric = PerceptualEvaluationSpeechQuality(fs=16000, mode="wb")
    stoi_metric = ShortTimeObjectiveIntelligibility(fs=16000, extended=False)
    sisdr_metric = ScaleInvariantSignalDistortionRatio()

    # Training params
    num_epochs = 50
    patience = 7
    best_val_pesq = -1.0
    no_improve_count = 0

    # For plotting
    train_losses, val_losses = [], []
    pesq_scores, stoi_scores, sisdr_scores = [], [], []

    for epoch in range(num_epochs):
        # ---------------- TRAIN ----------------
        model.train()
        epoch_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

        for clean, noisy, length in loop:
            clean = clean.to(device).squeeze(1)
            noisy = noisy.to(device).squeeze(1)
            optimizer.zero_grad()

            out_wave = model(noisy)

            # Combined loss (MSE + SI-SDR + Multi-Res STFT)
            loss = (
                0.6 * loss_fn(out_wave, clean)
                + 0.2 * si_sdr_loss(out_wave, clean)
                + 0.2 * multi_res_stft_loss(out_wave, clean)
            )
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = epoch_loss / len(train_loader)

        # ---------------- VALIDATION ----------------
        val_loss, val_pesq, val_stoi, val_sisdr = evaluate(
            model, val_loader, device, loss_fn, pesq_metric, stoi_metric, sisdr_metric
        )

        print(f"\nEpoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"PESQ: {val_pesq:.4f} | "
              f"STOI: {val_stoi:.4f} | "
              f"SI-SDR: {val_sisdr:.4f} dB")

        # Record values
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        pesq_scores.append(val_pesq)
        stoi_scores.append(val_stoi)
        sisdr_scores.append(val_sisdr)

        # ---------------- EARLY STOPPING ----------------
        if val_pesq > best_val_pesq:
            best_val_pesq = val_pesq
            no_improve_count = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("✅ Saved new best model (based on PESQ)")
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print("⏹ Early stopping triggered")
                break

        # Step the scheduler
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_pesq)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"🔽 LR reduced from {old_lr} to {new_lr}")

    print(f"\nTraining completed. Best PESQ: {best_val_pesq:.4f}")

    # ------------------- PLOTTING LEARNING CURVES -------------------
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(14, 8))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss", marker='o')
    plt.plot(epochs, val_losses, label="Val Loss", marker='o')
    plt.title("Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(epochs, pesq_scores, label="PESQ", marker='o', color='purple')
    plt.title("PESQ over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("PESQ")
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(epochs, stoi_scores, label="STOI", marker='o', color='green')
    plt.title("STOI over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("STOI")
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(epochs, sisdr_scores, label="SI-SDR", marker='o', color='orange')
    plt.title("SI-SDR over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("SI-SDR (dB)")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("learning_curves.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    train_model()

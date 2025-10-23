import os
import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.sdr import ScaleInvariantSignalDistortionRatio

from preprocess_DataLoader import load_data
from stft_utils import STFTProcessor
from tfdense_net import TFDense_Net
from tfdense_gan import TFDense_GAN


alpha = 0.5
beta = 0.5
lambda1, lambda2 = 1.0, 1.0

def compute_qprime(clean_mag, fake_mag):
    """
    Fast differentiable Q' proxy based on normalized magnitude correlation.
    clean_mag, fake_mag: [B, F, T]
    Returns: Q' ∈ [0, 1], shape [B,1,1,1]
    """
    eps = 1e-8
    dot = torch.sum(clean_mag * fake_mag, dim=(1,2))
    norm = torch.sqrt(torch.sum(clean_mag**2, dim=(1,2)) * torch.sum(fake_mag**2, dim=(1,2)) + eps)
    corr = (dot / (norm + eps)).clamp(0, 1)  # cosine-like similarity
    return corr.view(-1, 1, 1, 1)

# ------------------- Evaluation -------------------
def evaluate_gan(model, val_loader, device, stft_processor, loss_fn, stoi_metric, pesq_metric, sisdr_metric):
    """
    Evaluation loop for TFDenseGAN (only generator part).
    Computes TF-domain loss + waveform loss, and objective metrics.
    """
    model.generator.eval()
    val_loss = 0.0
    pesq_scores, stoi_scores, sisdr_scores = [], [], []

    with torch.no_grad():
        for clean, noisy, length in val_loader:
            noisy = noisy.to(device)  # [B, 1, T]
            clean = clean.to(device)  # [B, 1, T]

            # --- Forward pass through full model (handles STFT internally) ---
            denoised_wave, denoised_mag = model(noisy)

            # --- Compute clean magnitude for spectrogram loss ---
            clean_mag = stft_processor.compute_mag(clean)
            ###################################################################################################
            print("Pred stats:", denoised_mag.min().item(), denoised_mag.max().item(),
                  torch.isnan(denoised_mag).any().item(), torch.isinf(denoised_mag).any().item())  ## debug

            #denoised_mag = torch.clamp(denoised_mag, min=1e-8, max=1e2)            
            #denoised_wave = stft_processor.istft(denoised_mag, noisy_p, length=16000)
            

            # --- Spectrogram and waveform losses ---
            loss_m = loss_fn(denoised_mag, clean_mag)
            loss_w = loss_fn(denoised_wave, clean)
            loss = alpha * loss_w + beta * loss_m
            val_loss += loss.item()

            
            # --- Metrics ---
            est, ref = denoised_wave.cpu(), clean.cpu()
            try:
                stoi_scores.append(stoi_metric(est, ref).item())
                pesq_scores.append(pesq_metric(est, ref).item())
                sisdr_scores.append(sisdr_metric(est, ref).item())
            except Exception as e:
                print("Metric error:", e)

    print("denoised_wave stats:", denoised_wave.abs().mean().item(), denoised_wave.abs().max().item())

    # --- Aggregate results ---
    avg_loss = val_loss / len(val_loader)
    avg_stoi = sum(stoi_scores) / len(stoi_scores) if stoi_scores else 0.0
    avg_pesq = sum(pesq_scores) / len(pesq_scores) if pesq_scores else 0.0
    avg_sisdr = sum(sisdr_scores) / len(sisdr_scores) if sisdr_scores else 0.0

    # Reset metrics for next epoch
    stoi_metric.reset()
    pesq_metric.reset()
    sisdr_metric.reset()

    # Optional: monitor D outputs (diagnostics only)
    with torch.no_grad():
        fake_scores = model.discriminator(denoised_wav)
        real_scores = model.discriminator(clean)
        d_real_mean = sum(torch.mean(s).item() for s in real_scores) / len(real_scores)
        d_fake_mean = sum(torch.mean(s).item() for s in fake_scores) / len(fake_scores)

    print(f"[Eval] D(real): {d_real_mean:.3f}, D(fake): {d_fake_mean:.3f}")
    
    return avg_loss, avg_stoi, avg_pesq, avg_sisdr



def train_tfdense_gan():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Dataset + DataLoader
    dataset_dir = "../dataset/VCTK_DEMAND"
    train_loader, val_loader, test_loader = load_data(dataset_dir)

    # Model
    model = TFDense_GAN(device).to(device)
    stft_proc = STFTProcessor(device) 
    loss_fn = torch.nn.MSELoss()    

    # Separate optimizers
    opt_g = torch.optim.Adam(model.generator.parameters(), lr=10e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=10e-4, betas=(0.5, 0.999))

    # Metrics
    stoi_metric = ShortTimeObjectiveIntelligibility(fs=16000, extended=False)
    pesq_metric = PerceptualEvaluationSpeechQuality(fs=16000, mode="wb")
    sisdr_metric = ScaleInvariantSignalDistortionRatio()

    os.makedirs("checkpoint_tfdensegan", exist_ok=True)

    num_epochs=2

    for epoch in range(num_epochs):
        model.train()
        epoch_g_loss, epoch_d_loss = 0.0, 0.0

        pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for clean_wave, noisy_wave, _ in pbar:
            noisy_wave = noisy_wave.to(device)
            clean_wave = clean_wave.to(device)

            # ---------------------------------------
            # 1. Forward pass through Generator
            # ---------------------------------------
            denoised_wave, denoised_mag = model(noisy_wave)   # G(x) returns denoised_wav and denoised_mag
            clean_mag, noisy_mag, noisy_phase = stft_proc.stft(clean_wave, noisy_wave)

            # ---------------------------------------
            # Spectrogram and waveform losses --> TF Loss
            # ---------------------------------------
            loss_spect = loss_fn(denoised_mag, clean_mag)
            loss_wave = loss_fn(denoised_wave, clean_wave)
            loss_tf = alpha * loss_wave + beta * loss_spect

            # ---------------------------------------
            # Adversarial loss (Generator)
            # ---------------------------------------
            fake_scores = model.discriminator(denoised_wave)
            loss_gan = 0
            for score in fake_scores:
                loss_gan += torch.mean((score - 1) ** 2)

            loss_gan /= len(fake_scores)  # average across discriminators

            # ---------------------------------------
            # Total Generator Loss
            # ---------------------------------------
            loss_g_total = lambda1*loss_tf + lambda2*loss_gan

            opt_g.zero_grad()
            loss_g_total.backward()
            torch.nn.utils.clip_grad_norm_(model.generator.parameters(), 5.0)
            opt_g.step()

            # ---------------------------------------
            # Adversarial loss (Discriminator)
            # ---------------------------------------
            qprime = compute_qprime(clean_mag, denoised_mag.detach())
            real_scores = model.discriminator(clean_wave)
            fake_scoress = model.discriminator(denoised_wave.detach())            

            loss_d = 0
            for D_real, D_fake in zip(real_scores, fake_scoress):
                term_real = torch.mean((D_real - 1) ** 2)
                term_fake = torch.mean((D_fake - qprime) ** 2)
                loss_d += term_real + term_fake

            loss_d /= len(real_scores)

            opt_d.zero_grad()
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), 5.0)
            opt_d.step()

            # ---------------------------------------
            # Logging
            # ---------------------------------------
            epoch_g_loss += loss_g_total.item()
            epoch_d_loss += loss_d.item()
            pbar.set_postfix({
                "L_G": f"{loss_g_total.item():.4f}",
                "L_D": f"{loss_d.item():.4f}"
            })

        # ---------------------------------------
        # Save checkpoint
        # ---------------------------------------
        ckpt_path = f"checkpoint_tfdensegan/epoch_{epoch+1}.pth"
        torch.save({
            "epoch": epoch + 1,
            "generator_state_dict": model.generator.state_dict(),
            "discriminator_state_dict": model.discriminator.state_dict(),
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict(),
            "loss_g": epoch_g_loss / len(train_loader),
            "loss_d": epoch_d_loss / len(train_loader)
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

        # ---------------------------------------
        # Validation metrics
        # ---------------------------------------
        val_loss, val_stoi, val_pesq, val_sisdr = evaluate_gan(
            model, val_loader, device, stft_proc,
            loss_fn, stoi_metric, pesq_metric, sisdr_metric
        )
        print(f"\nEpoch [{epoch+1}/{num_epochs}] "
              f"--> Avg G Loss: {epoch_g_loss/len(train_loader):.6f} "
              f"--> Avg D Loss: {epoch_d_loss/len(train_loader):.6f}\n"
              f"--> Val Loss: {val_loss:.6f}\t"
              f"--> STOI: {val_stoi:.4f}\t"
              f"--> PESQ: {val_pesq:.4f}\t"
              f"--> SI-SDR: {val_sisdr:.4f} dB")

    print("Training complete!")



if __name__ == "__main__":
    train_tfdense_gan()

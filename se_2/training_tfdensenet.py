import csv
import os
import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.sdr import ScaleInvariantSignalDistortionRatio

from torchsummary import summary

from preprocess import load_traindata_with_subset_sampler
from stft_utils import STFTProcessor
from tfdense_net import TFDense_Net
from scheduler import DynamicLRScheduler

alpha = 1
beta = 0.01

Mean = -0.00020485175894189195
Var = 0.004213959431448535
Std = 0.06491501699490292


# ------------------- Evaluation -------------------
def evaluate(model, val_loader, device, stft_processor, loss_fn, stoi_metric, pesq_metric, sisdr_metric):
    model.eval()
    val_loss = 0.0
    pesq_scores, stoi_scores, sisdr_scores = [], [], []

    with torch.no_grad():
    
        pbar = tqdm(val_loader, desc=f"Validation ...") 
        for clean, noisy, length in val_loader:

            noisy = noisy.to(device)  # [B, 1, T]
            clean = clean.to(device)  # [B, 1, T]
            
            clean_norm = (clean - Mean) / Std
            noisy_norm = (noisy - Mean) / Std

            # --- Forward pass ---
            clean_m, noisy_m, noisy_p = stft_processor.stft(clean_norm, noisy_norm)	# [B, F, T]

            # Predict enhanced magnitude from noisy magnitude
            pred_m = model(noisy_m.unsqueeze(1).permute(0,1,3,2)) 
            pred_m = pred_m.squeeze(1).permute(0,2,1)              # back to [B, F, T]

            if pred_m.shape != clean_m.shape:
                print("Predicted shape mismatch!\n")

            # --- Loss in spectrogram domain ---
            loss_m = loss_fn(pred_m, clean_m)

            # --- Reconstruct waveform ---
            pred_m = torch.clamp(pred_m, min=1e-8, max=1e2)            
            rec_wav = stft_processor.istft(pred_m, noisy_p, length=16000)

            # --- Loss in waveform domain ---
            loss_wav = loss_fn(rec_wav, clean_norm)

            # Total loss
            loss = alpha*loss_wav + beta*loss_m
            val_loss += loss.item()
            
            # Denormalize to original waveform domain
            rec_wav = rec_wav * Std + Mean 

            est, ref = rec_wav.cpu(), clean.cpu()
            try:
                stoi_scores.append(stoi_metric(est, ref).item())
                pesq_scores.append(pesq_metric(est, ref).item())
                sisdr_scores.append(sisdr_metric(est, ref).item())
            except Exception as e:
                print("Metric error:", e)
   
    avg_loss = val_loss / len(val_loader)
    avg_stoi = sum(stoi_scores) / len(stoi_scores) if stoi_scores else 0.0
    avg_pesq = sum(pesq_scores) / len(pesq_scores) if pesq_scores else 0.0
    avg_sisdr = sum(sisdr_scores) / len(sisdr_scores) if sisdr_scores else 0.0

    # reset metrics
    stoi_metric.reset()
    pesq_metric.reset()
    sisdr_metric.reset()

    return avg_loss, avg_stoi, avg_pesq, avg_sisdr


#-------------------- Training ---------------------
def train_tfdensenet():
    
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "training_log.csv")
    
    # Create CSV file with header
    if not os.path.exists(log_path):
        with open(log_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "val_stoi", "val_pesq", "val_sisdr"])

    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Dataset + DataLoader
    dataset_dir = "../dataset/VCTK_DEMAND_DNS_datasets"

    # Check if exists
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset not found at {dataset_dir}")
    
    #train_loader, val_loader, test_loader = load_data(dataset_dir)

    # Model, loss, optimizer
    model = TFDense_Net().to(device)
    stft_processor = STFTProcessor(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))
    scheduler = DynamicLRScheduler(optimizer, d_model=64, warmup_steps=4000, k1=0.2, k2=4e-4)

    # Metrics
    stoi_metric = ShortTimeObjectiveIntelligibility(fs=16000, extended=False)
    pesq_metric = PerceptualEvaluationSpeechQuality(fs=16000, mode="wb")
    sisdr_metric = ScaleInvariantSignalDistortionRatio()
    
    # Training params
    start_epoch = 0
    num_epochs = 300

    # --- Checkpoint handling ---
    checkpoint_dir = "checkpoint"
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]

    # Auto-resume: load latest checkpoint if present
    if checkpoint_files:
        latest_ckpt = max(
            checkpoint_files,
            key=lambda f: int(f.split("_")[1].split(".")[0])  # assumes format epoch_XX.pth
        )
        ckpt_path = os.path.join(checkpoint_dir, latest_ckpt)
        checkpoint = torch.load(ckpt_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        print(f"Resumed from checkpoint: {ckpt_path} (epoch {start_epoch})")
    else:
        print("Starting training from epoch 0.")

    
    # Model summary
    # summary(model, input_size=(1, 257, 63))
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        train_loader, val_loader = load_traindata_with_subset_sampler(dataset_dir)
        
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") 
        for clean_wave, noisy_wave, _ in pbar:
            noisy_wave = noisy_wave.to(device)  # [B, 1, T]
            clean_wave = clean_wave.to(device)  # [B, 1, T]
            
            # Apply global waveform normalization
            clean_wave = (clean_wave - Mean) / Std
            noisy_wave = (noisy_wave - Mean) / Std

            # --- Forward pass ---
            clean_mag, noisy_mag, noisy_phase = stft_processor.stft(clean_wave, noisy_wave)

            # Predict enhanced magnitude from noisy magnitude
            pred_mag = model(noisy_mag.unsqueeze(1).permute(0,1,3,2)) 
            pred_mag = pred_mag.squeeze(1).permute(0,2,1)            # back to [B, F, T]

            if pred_mag.shape != clean_mag.shape:
                print("Predicted shape mismatch!\n")

            # --- Loss in spectrogram domain ---       
            loss_mag = loss_fn(pred_mag, clean_mag)

            # --- Reconstruct waveform ---
            pred_mag = torch.clamp(pred_mag, min=1e-8, max=1e2)
            rec_wave = stft_processor.istft(pred_mag, noisy_phase, length=16000)            

            # --- Loss in waveform domain ---
            loss_wave = loss_fn(rec_wave, clean_wave)
            
            # Total loss
            loss = alpha*loss_wave + beta*loss_mag

            if torch.isnan(loss):
                print("NaN loss detected, skipping step")
                continue 

            # --- Backprop ---
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (L2-norm max = 5)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step() 

            epoch_loss += loss.item()
            
            pbar.set_postfix({"Batch_loss": loss.item(),"Spec_loss": loss_mag.item(),"Wav_loss": loss_wave.item(),})

        avg_loss = epoch_loss / len(train_loader)
        #avg_spec_loss = epoch_spec_loss / len(train_loader)
        #avg_wav_loss = epoch_wav_loss / len(train_loader)
       
        # Save checkpoint every 1 epochs        
        if (epoch + 1) % 1 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pth")
            torch.save({
                "epoch": (epoch+1),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
 
        scheduler.epoch_step()           
          
        # ---------------- VALIDATION ----------------
        val_loss, val_stoi, val_pesq, val_sisdr = evaluate(
            model, val_loader, device, stft_processor, loss_fn, stoi_metric, pesq_metric, sisdr_metric
        )

        print(f"\nEpoch [{epoch+1}/{num_epochs}] "
              f"-->Avg Train Loss: {avg_loss:.6f}\t"
              f"--> Val Loss: {val_loss:.6f}\t"
              f"--> STOI: {val_stoi:.4f}\t"
              f"--> PESQ: {val_pesq:.4f}\t"
              f"--> SI-SDR: {val_sisdr:.4f} dB\n")
        
        try:
            with open(log_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, avg_loss, val_loss, val_stoi, val_pesq, val_sisdr])
        except Exception as e:
            print(f"Failed to log epoch {epoch+1}: {e}")
        

if __name__ == "__main__":
    train_tfdensenet()

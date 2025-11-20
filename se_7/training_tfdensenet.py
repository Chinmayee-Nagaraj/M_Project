import csv
import os
import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.sdr import ScaleInvariantSignalDistortionRatio

from preprocess import load_traindata_with_subset_sampler
from stft_utils import STFTProcessor
from tfdense_net import TFDense_Net
from scheduler import DynamicLRScheduler
from loss_utils import SISDRLoss

torch.cuda.empty_cache()

alpha = 1.0      # waveform loss weight
beta = 1.0       # spectral loss weight 
gamma = 0.1      # SI-SDR loss weight

# ------------------- Evaluation -------------------
def evaluate(model, val_loader, device, stft_processor, sisdr_loss_fn, stoi_metric, pesq_metric, sisdr_metric):
    model.eval()
    val_loss = 0.0
    pesq_scores, stoi_scores, sisdr_scores = [], [], []

    loss_wav_sum = 0.0
    loss_spect_sum = 0.0
    loss_sisdr_sum = 0.0

    with torch.no_grad():
    
        pbar = tqdm(val_loader, desc=f"Validation ...") 
        for clean, noisy, length in pbar:
            L = clean.size(-1)       

            noisy = noisy.to(device)  # [B, 1, T]
            clean = clean.to(device)  # [B, 1, T]
            
            clean_mean = clean.mean(dim=-1, keepdim=True)
            clean_std  = clean.std(dim=-1, keepdim=True) + 1e-8
            clean_norm = (clean - clean_mean) / clean_std
            
            noisy_mean = noisy.mean(dim=-1, keepdim=True)
            noisy_std  = noisy.std(dim=-1, keepdim=True) + 1e-8
            noisy_norm = (noisy - noisy_mean) / noisy_std

            # --- Forward pass ---
            clean_r, clean_i, noisy_r, noisy_i = stft_processor.stft_ri(clean_norm, noisy_norm)	# [B, F, T]
            
            x = torch.stack([noisy_r, noisy_i], dim=1)	#[B, 2, F, T] 

            # Predict mask
            pred_mask = model(x)  # [B, 2, F, T] - predicted mask
            
            # Apply mask to noisy spectrum
            pred_r = pred_mask[:, 0, :, :] * noisy_r  # [B, F, T]
            pred_i = pred_mask[:, 1, :, :] * noisy_i  # [B, F, T]

            # --- Loss in spectrogram domain ---
            num_tf_elements = pred_r.numel()
            loss_r = torch.nn.functional.mse_loss(pred_r, clean_r)
            loss_i = torch.nn.functional.mse_loss(pred_i, clean_i)
            loss_spect = (loss_r + loss_i)  # Already normalized by MSE

            # --- Reconstruct waveform ---
            pred_complex = torch.complex(pred_r, pred_i)
            rec_wav = stft_processor.istft_ri(pred_complex, length=L)            

            # --- Loss in waveform domain ---
            loss_wav = torch.nn.functional.mse_loss(rec_wav, clean_norm)

            # Denormalize the rec_wave
            rec_wav_denorm = rec_wav * clean_std + clean_mean

            # SISDR Loss  
            loss_sisdr = sisdr_loss_fn(rec_wav_denorm, clean)  
                    
            # Total Loss            
            loss = alpha*loss_wav + beta*loss_spect + gamma*loss_sisdr

            if torch.isnan(loss):
                print("NaN loss detected, skipping step")
                continue 

            val_loss += float(loss)
            loss_wav_sum += float(loss_wav)
            loss_spect_sum += float(loss_spect)
            loss_sisdr_sum += float(loss_sisdr)
 
            est = rec_wav_denorm.detach().cpu()
            ref = clean.detach().cpu()

            try:
                stoi_scores.append(float(stoi_metric(est, ref)))
                pesq_scores.append(float(pesq_metric(est, ref)))
                sisdr_scores.append(float(sisdr_metric(est, ref)))
            except Exception as e:
                print("Metric error:", e)
   
    n_batches = len(val_loader)
    avg_loss = val_loss / n_batches
    avg_loss_wav = loss_wav_sum / n_batches
    avg_loss_spect = loss_spect_sum / n_batches
    avg_loss_sisdr = loss_sisdr_sum / n_batches

    avg_stoi = sum(stoi_scores) / len(stoi_scores) if stoi_scores else 0.0
    avg_pesq = sum(pesq_scores) / len(pesq_scores) if pesq_scores else 0.0
    avg_sisdr = sum(sisdr_scores) / len(sisdr_scores) if sisdr_scores else 0.0

    # reset metrics
    stoi_metric.reset()
    pesq_metric.reset()
    sisdr_metric.reset()

    return avg_loss, avg_stoi, avg_pesq, avg_sisdr, avg_loss_wav, avg_loss_spect, avg_loss_sisdr

#-------------------- Training ---------------------
def train_tfdensenet():
    
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "training_log.csv")
    
    # Create CSV file with header
    if not os.path.exists(log_path):
        with open(log_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_wav", "train_spect", "train_sisdr",
                           "val_loss", "val_wav", "val_spect", "val_sisdr", 
                           "val_stoi", "val_pesq", "val_sisdr_metric"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loss weights - Alpha: {alpha}, Beta: {beta}, Gamma: {gamma}")

    # Dataset + DataLoader
    dataset_dir = "../dataset/VCTK_DEMAND_DNS_datasets"

    # Check if exists
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset not found at {dataset_dir}")
    
    # Model
    model = TFDense_Net().to(device)
    stft_processor = STFTProcessor(device)
    sisdr_loss_fn = SISDRLoss()

    # Optimizer with weight decay for regularization
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99), weight_decay=1e-5)
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


    # Training loop
    for epoch in range(start_epoch, num_epochs):
        train_loader, val_loader = load_traindata_with_subset_sampler(dataset_dir)
        
        model.train()
        epoch_loss = 0.0
        epoch_loss_wav = 0.0
        epoch_loss_spect = 0.0
        epoch_loss_sisdr = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") 
        for clean_wave, noisy_wave, _ in pbar:
            Length = clean_wave.size(-1) 
                   
            noisy_wave = noisy_wave.to(device)  # [B, 1, T]
            clean_wave = clean_wave.to(device)  # [B, 1, T]
            
            # Apply waveform normalization
            clean_mean = clean_wave.mean(dim=-1, keepdim=True)
            clean_std  = clean_wave.std(dim=-1, keepdim=True) + 1e-8
            clean_norm = (clean_wave - clean_mean) / clean_std
            
            noisy_mean = noisy_wave.mean(dim=-1, keepdim=True)
            noisy_std  = noisy_wave.std(dim=-1, keepdim=True) + 1e-8
            noisy_wave = (noisy_wave - noisy_mean) / noisy_std

            # --- Forward pass ---
            clean_real, clean_imag, noisy_real, noisy_imag = stft_processor.stft_ri(clean_norm, noisy_wave)
            
            x = torch.stack([noisy_real, noisy_imag], dim=1)	#[B, 2, F, T] 

            # Predict mask
            pred_mask = model(x)  # [B, 2, F, T] - predicted mask
            
            # Apply mask to noisy spectrum (element-wise multiplication)
            pred_real = pred_mask[:, 0, :, :] * noisy_real  # [B, F, T]
            pred_imag = pred_mask[:, 1, :, :] * noisy_imag  # [B, F, T]

            # Optional sanity checks (uncomment while debugging)
            assert pred_real.shape == clean_real.shape
            assert pred_imag.shape == clean_imag.shape

            # --- Loss in spectrum ---       
            loss_real = torch.nn.functional.mse_loss(pred_real, clean_real)
            loss_imag = torch.nn.functional.mse_loss(pred_imag, clean_imag)
            loss_spect = (loss_real + loss_imag)

            # --- Reconstruct waveform ---
            pred_complex = torch.complex(pred_real, pred_imag)
            rec_wave = stft_processor.istft_ri(pred_complex, length=Length) 

            # --- Loss in waveform domain ---
            loss_wave = torch.nn.functional.mse_loss(rec_wave, clean_norm)
                        
            # Denormalize the rec_wave
            rec_wave_denorm = rec_wave * clean_std + clean_mean

            # SISDR Loss  
            loss_sisdr = sisdr_loss_fn(rec_wave_denorm, clean_wave)  
                    
            # Total Loss            
            loss = alpha*loss_wave + beta*loss_spect + gamma*loss_sisdr

            if torch.isnan(loss):
                print("NaN loss detected, skipping step")
                print(f"  loss_wave: {loss_wave}, loss_spect: {loss_spect}, loss_sisdr: {loss_sisdr}")
                continue 

            # --- Backprop ---
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (L2-norm max = 5)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            lr = scheduler.step() 
            optimizer.step()

            epoch_loss += float(loss)
            epoch_loss_wav += float(loss_wave)
            epoch_loss_spect += float(loss_spect)
            epoch_loss_sisdr += float(loss_sisdr)
            
            pbar.set_postfix({
                "Total": f"{float(loss):.4f}",
                "Wave": f"{float(loss_wave):.4f}",
                "Spect": f"{float(loss_spect):.4f}",
                "-SiSDR": f"{float(loss_sisdr):.4f}",
                "LR": f"{lr:.6f}"
            })

        avg_loss = epoch_loss / len(train_loader)
        avg_loss_wav = epoch_loss_wav / len(train_loader)
        avg_loss_spect = epoch_loss_spect / len(train_loader)
        avg_loss_sisdr = epoch_loss_sisdr / len(train_loader)
       
        # Save checkpoint every 1 epochs        
        if (epoch + 1) % 1 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pth")
            torch.save({
                "epoch": (epoch+1),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }, checkpoint_path)
            print(f"\nSaved checkpoint: {checkpoint_path}")
 
        scheduler.epoch_step()           
          
        # ---------------- VALIDATION ----------------
        val_loss, val_stoi, val_pesq, val_sisdr_metric, val_loss_wav, val_loss_spect, val_loss_sisdr = evaluate(
                model, val_loader, device, stft_processor, sisdr_loss_fn, 
                stoi_metric, pesq_metric, sisdr_metric
        )

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train - Total: {avg_loss:.4f} | Wave: {avg_loss_wav:.4f} | Spect: {avg_loss_spect:.4f} | -SiSDR: {avg_loss_sisdr:.4f}")
        print(f"  Val   - Total: {val_loss:.4f} | Wave: {val_loss_wav:.4f} | Spect: {val_loss_spect:.4f} | -SiSDR: {val_loss_sisdr:.4f}")
        print(f"  Metrics - STOI: {val_stoi:.4f} | PESQ: {val_pesq:.4f} | SI-SDR: {val_sisdr_metric:.2f} dB")

        try:
            with open(log_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, avg_loss, avg_loss_wav, avg_loss_spect, avg_loss_sisdr,
                               val_loss, val_loss_wav, val_loss_spect, val_loss_sisdr,
                               val_stoi, val_pesq, val_sisdr_metric])
        except Exception as e:
            print(f"Failed to log epoch {epoch+1}: {e}")
        

if __name__ == "__main__":
    train_tfdensenet()

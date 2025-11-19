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

alpha = 1
beta = 0.005
gamma = 0.001

# ------------------- Evaluation -------------------
def evaluate(model, val_loader, device, stft_processor, loss_fn, sisdr_loss_fn, stoi_metric, pesq_metric, sisdr_metric):
    model.eval()
    val_loss = 0.0
    pesq_scores, stoi_scores, sisdr_scores = [], [], []

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

            # Predict enhanced magnitude from noisy magnitude
            pred = model(x) 
            pred_r = pred[:, 0, :, :]	# back to [B, F, T]
            pred_i = pred[:, 1, :, :]  # back to [B, F, T]

            # --- Loss in spectrogram domain ---
            loss_r = loss_fn(pred_r, clean_r)
            loss_i = loss_fn(pred_i, clean_i)
            
            loss_spect = loss_r + loss_i

            # --- Reconstruct waveform ---
            pred_complex = torch.complex(pred_r, pred_i)
            rec_wav = stft_processor.istft_ri(pred_complex, length=L)            

            # --- Loss in waveform domain ---
            loss_wav = loss_fn(rec_wav, clean_norm)
            
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
 
            est = rec_wav_denorm.detach().cpu()
            ref = clean.detach().cpu()

            try:
                stoi_scores.append(float(stoi_metric(est, ref)))
                pesq_scores.append(float(pesq_metric(est, ref)))
                sisdr_scores.append(float(sisdr_metric(est, ref)))
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
    sisdr_loss_fn = SISDRLoss()
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


    # Training loop
    for epoch in range(start_epoch, num_epochs):
        train_loader, val_loader = load_traindata_with_subset_sampler(dataset_dir)
        
        model.train()
        epoch_loss = 0.0
        
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

            # Predict enhanced magnitude from noisy magnitude
            pred = model(x) 
            pred_real = pred[:, 0, :, :]	# back to [B, F, T]
            pred_imag = pred[:, 1, :, :]      # back to [B, F, T]
            
            # Optional sanity checks (uncomment while debugging)
            assert pred_real.shape == clean_real.shape
            assert pred_imag.shape == clean_imag.shape

            # --- Loss in spectrum ---       
            loss_real = loss_fn(pred_real, clean_real)
            loss_imag = loss_fn(pred_imag, clean_imag)
            
            loss_spect = loss_real + loss_imag

            # --- Reconstruct waveform ---
            pred_complex = torch.complex(pred_real, pred_imag)
            rec_wave = stft_processor.istft_ri(pred_complex, length=Length) 

            # --- Loss in waveform domain ---
            loss_wave = loss_fn(rec_wave, clean_norm)
            
            # Denormalize the rec_wave
            rec_wave_denorm = rec_wave * clean_std + clean_mean

            # SISDR Loss  
            loss_sisdr = sisdr_loss_fn(rec_wave_denorm, clean_wave)  
                    
            # Total Loss            
            loss = alpha*loss_wave + beta*loss_spect + gamma*loss_sisdr

            if torch.isnan(loss):
                print("NaN loss detected, skipping step")
                continue 

            # --- Backprop ---
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (L2-norm max = 5)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            lr = scheduler.step() 
            optimizer.step()

            epoch_loss += float(loss)
            
            pbar.set_postfix({
            	"Total": float(loss),
        	"Wave": float(loss_wave),            	
            	"Spect": float(loss_spect),
            	"SiSDR": float(loss_sisdr)})


        avg_loss = epoch_loss / len(train_loader)
       
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
            model, val_loader, device, stft_processor, loss_fn, sisdr_loss_fn, stoi_metric, pesq_metric, sisdr_metric
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

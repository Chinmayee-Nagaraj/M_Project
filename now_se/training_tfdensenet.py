import os
import csv
from tqdm import tqdm

import torch
import torchaudio
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.sdr import ScaleInvariantSignalDistortionRatio

from dataloader import load_datasets
from tfdense_net import TFDense_Net
from stft_utils import STFTProcessor
from scheduler import DynamicLRScheduler
from loss_utils import SISDRLoss, MultiResolutionSTFTLoss
import warnings

torch.cuda.empty_cache()

# Loss weights
alpha_wave  = 2000.0
alpha_spect = 10.0
alpha_sisdr = 0.01
#alpha_stft  = 0.08

####################################################
#-------------------- Evaluation ---------------------
####################################################
def evaluate(model, val_loader, device, stft_processor, sisdr_loss_fn, stft_loss_fn, stoi_metric, pesq_metric, sisdr_metric):
    model.eval()
    val_loss = 0.0
    loss_wav_sum = 0.0
    loss_spect_sum = 0.0
    loss_sisdr_sum = 0.0
    loss_stft_sum = 0.0
    pesq_scores, stoi_scores, sisdr_scores = [], [], []
    
    num_samples = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Validation...") 
        for batch in pbar:  # batch is a list of samples
            num_samples += len(batch)
            for clean_wave, noisy_wave, length, fname in batch:  

                clean_wave = clean_wave.unsqueeze(0).to(device)  # [1, 1, T]
                noisy_wave = noisy_wave.unsqueeze(0).to(device)  # [1, 1, T]
            
                clean_real, clean_imag, noisy_real, noisy_imag = stft_processor.stft_ri(clean_wave, noisy_wave)
                x = torch.stack([noisy_real, noisy_imag], dim=1)	#[1, 2, F, T] 
    
                # --- Forward pass ---
                pred_mask = model(x)  # [1, 2, F, T] predicted mask

                # Split predicted mask
                M_real = pred_mask[:, 0, :, :]   # [B, F, T]
                M_imag = pred_mask[:, 1, :, :]   # [B, F, T]

                # Apply complex mask
                pred_real = (M_real * noisy_real) - (M_imag * noisy_imag)
                pred_imag = (M_real * noisy_imag) + (M_imag * noisy_real)

                # --- Reconstruct to get predicted waveform ---
                pred_complex = torch.complex(pred_real, pred_imag)
                pred_wave = stft_processor.istft_ri(pred_complex, length=length) 
    
                # --- Loss in waveform domain ---
                loss_wave = torch.nn.functional.mse_loss(pred_wave, clean_wave)
    
                # --- Loss in spectrum ---       
                loss_real = torch.nn.functional.mse_loss(pred_real, clean_real)
                loss_imag = torch.nn.functional.mse_loss(pred_imag, clean_imag)
                loss_spect = (loss_real + loss_imag)
    
                # SISDR Loss  
                loss_sisdr = sisdr_loss_fn(clean_wave, pred_wave) 
    
                # Multi-Resolution STFT Loss
                loss_stft = stft_loss_fn(clean_wave, pred_wave)
                    
                # Total Loss 
                loss = (alpha_wave * loss_wave +
                        alpha_spect * loss_spect +
                        alpha_sisdr * loss_sisdr ) # + alpha_stft * loss_stft)

                val_loss += float(loss)
                loss_wav_sum += float(loss_wave)
                loss_spect_sum += float(loss_spect)
                loss_sisdr_sum += float(loss_sisdr)
                loss_stft_sum += float(loss_stft) 
 
                est = pred_wave.detach().cpu()
                ref = clean_wave.detach().cpu()
    
                try:
                    stoi_scores.append(float(stoi_metric(est, ref)))
                    pesq_scores.append(float(pesq_metric(est, ref)))
                    sisdr_scores.append(float(sisdr_metric(est, ref)))
                except Exception as e:
                    print("Metric error:", e)
   
        avg_loss = val_loss / num_samples
        avg_loss_wav = loss_wav_sum / num_samples
        avg_loss_spect = loss_spect_sum / num_samples
        avg_loss_sisdr = loss_sisdr_sum / num_samples
        avg_loss_stft = loss_stft_sum / num_samples

        avg_stoi = sum(stoi_scores) / len(stoi_scores) if stoi_scores else 0.0
        avg_pesq = sum(pesq_scores) / len(pesq_scores) if pesq_scores else 0.0
        avg_sisdr = sum(sisdr_scores) / len(sisdr_scores) if sisdr_scores else 0.0

        # reset metrics
        stoi_metric.reset()
        pesq_metric.reset()
        sisdr_metric.reset()

        return avg_loss, avg_stoi, avg_pesq, avg_sisdr, avg_loss_wav, avg_loss_spect, avg_loss_sisdr, avg_loss_stft


####################################################
#-------------------- Training ---------------------
####################################################

def train_tfdensenet():
    
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "training_log.csv")
    
    # Create CSV file with header
    if not os.path.exists(log_path):
        with open(log_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_wav", "train_spect", "train_sisdr", "train_stft",
                           "val_loss", "val_wav", "val_spect", "val_sisdr", "val_stft",
                           "val_stoi", "val_pesq", "val_sisdr_metric"]) 

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Dataset root directory
    dataset_dir = "../dataset/Subset_Dataset"
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset not found at {dataset_dir}")

    # Dataloading
    train_loader, val_loader, test_loader = load_datasets(root_dir=dataset_dir, batch_size=32) 
    
    # Model
    model = TFDense_Net().to(device)

    # STFT Processor
    stft_processor = STFTProcessor(device)

    # Loss Functions
    sisdr_loss_fn = SISDRLoss()
    stft_loss_fn = MultiResolutionSTFTLoss(device=device)

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

    # -------- Checkpoint handling --------
    checkpoint_dir = "checkpoint"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Auto-resume: load latest checkpoint if present
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]

    if checkpoint_files:
        latest_ckpt = max(
            checkpoint_files,
            key=lambda f: int(f.split("_")[1].split(".")[0])  # assumes format epoch_XX.pth
        )
        ckpt_path = os.path.join(checkpoint_dir, latest_ckpt)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

        # Load states
        model.load_state_dict(checkpoint["model_state_dict"])        
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = checkpoint.get("epoch", 0)
        last_loss = checkpoint.get("loss", None)
        print(f"Resumed from checkpoint: {ckpt_path} (epoch {start_epoch})")

    else:
        print("Starting training from epoch 0")

    print("Loss weights: alpha_wave = {} | alpha_spect = {} | alpha_sisdr = {}".format(alpha_wave, alpha_spect, alpha_sisdr))


    # -------- Training loop --------
    for epoch in range(start_epoch, num_epochs):
        
        model.train()
        epoch_loss = 0.0
        epoch_loss_wav = 0.0
        epoch_loss_spect = 0.0
        epoch_loss_sisdr = 0.0
        epoch_loss_stft = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") 
        for clean_wave, noisy_wave, L, _ in pbar:
            Length = clean_wave.size(-1)  
            
            optimizer.zero_grad()
            
            clean_wave = clean_wave.to(device)  # [B, 1, T]
            noisy_wave = noisy_wave.to(device)  # [B, 1, T]
            
            clean_real, clean_imag, noisy_real, noisy_imag = stft_processor.stft_ri(clean_wave, noisy_wave)
            x = torch.stack([noisy_real, noisy_imag], dim=1)	#[B, 2, F, T] 

            # --- Forward pass ---
            pred_mask = model(x)  # [B, 2, F, T] predicted mask

            # Split predicted mask
            M_real = pred_mask[:, 0, :, :]   # [B, F, T]
            M_imag = pred_mask[:, 1, :, :]   # [B, F, T]

            # Apply complex mask
            pred_real = (M_real * noisy_real) - (M_imag * noisy_imag)
            pred_imag = (M_real * noisy_imag) + (M_imag * noisy_real)

            # --- Reconstruct to get predicted waveform ---
            pred_complex = torch.complex(pred_real, pred_imag)
            pred_wave = stft_processor.istft_ri(pred_complex, length=Length) 

            # --- Loss in waveform domain ---
            loss_wave = torch.nn.functional.mse_loss(pred_wave, clean_wave)

            # --- Loss in spectrum ---       
            loss_real = torch.nn.functional.mse_loss(pred_real, clean_real)
            loss_imag = torch.nn.functional.mse_loss(pred_imag, clean_imag)
            loss_spect = (loss_real + loss_imag)

            # SISDR Loss  
            loss_sisdr = sisdr_loss_fn(clean_wave, pred_wave) 

            # Multi-Resolution STFT Loss
            loss_stft = stft_loss_fn(clean_wave, pred_wave)
                    
            # Total Loss 
            loss = (alpha_wave * loss_wave +
                    alpha_spect * loss_spect +
                    alpha_sisdr * loss_sisdr ) # + alpha_stft * loss_stft)

            # --- Backprop ---
            loss.backward()

            # Gradient clipping (L2-norm max = 5)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            lr = scheduler.step() 
            optimizer.step()

            epoch_loss += float(loss)
            epoch_loss_wav += float(loss_wave)
            epoch_loss_spect += float(loss_spect)
            epoch_loss_sisdr += float(loss_sisdr)
            epoch_loss_stft += float(loss_stft)
            
            pbar.set_postfix({
                "Total": f"{float(loss):.4f}",
                "Wave": f"{float(loss_wave):.4f}",
                "Spect": f"{float(loss_spect):.4f}",
                "-SiSDR": f"{float(loss_sisdr):.4f}",
                "STFT": f"{float(loss_stft):.4f}",
                "LR": f"{lr:.6f}"
            })
        
        total_len = len(train_loader)
        avg_loss = epoch_loss / total_len
        avg_loss_wav = epoch_loss_wav / total_len
        avg_loss_spect = epoch_loss_spect / total_len
        avg_loss_sisdr = epoch_loss_sisdr / total_len
        avg_loss_stft = epoch_loss_stft / total_len
      
       
        # Save checkpoint every 1 epochs     
        if (epoch + 1) % 1 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pth")
            torch.save({
                "epoch": (epoch+1),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": avg_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
 
        scheduler.epoch_step()           
          
        # ---------------- VALIDATION ----------------
        val_loss, val_stoi, val_pesq, val_sisdr_metric, val_loss_wav, val_loss_spect, val_loss_sisdr, val_loss_stft = evaluate(
                model, val_loader, device, stft_processor, sisdr_loss_fn, stft_loss_fn,
                stoi_metric, pesq_metric, sisdr_metric
        )

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train - Total: {avg_loss:.4f}| Wave: {avg_loss_wav:.4f}| Spect: {avg_loss_spect:.4f}| -SiSDR: {avg_loss_sisdr:.4f}| STFT: {avg_loss_stft:.4f}")
        print(f"  Val   - Total: {val_loss:.4f}| Wave: {val_loss_wav:.4f}| Spect: {val_loss_spect:.4f}| -SiSDR: {val_loss_sisdr:.4f}| STFT: {val_loss_stft:.4f}")
        print(f"  Metrics - STOI: {val_stoi:.4f} | PESQ: {val_pesq:.4f} | SI-SDR: {val_sisdr_metric:.2f} dB\n")

        try:
            with open(log_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, avg_loss, avg_loss_wav, avg_loss_spect, avg_loss_sisdr, avg_loss_stft,
                               val_loss, val_loss_wav, val_loss_spect, val_loss_sisdr, val_loss_stft,
                               val_stoi, val_pesq, val_sisdr_metric])
        except Exception as e:
            print(f"Failed to log epoch {epoch+1}: {e}")
        

if __name__ == "__main__":
    train_tfdensenet()

import argparse
import os
import sys
import time
import warnings

# Suppress most warnings for cleaner logs (comment out if debugging is needed)
warnings.filterwarnings("ignore")

import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler

from utils.general_utils import (
load_config,
load_and_prepare_data,
create_dataloader,
save_checkpoint,
load_checkpoint,
normalize_tensor_to_zero_one,
)

from utils.utils_fm import (
build_model,
sample_with_solver,
#calculate_metrics,
save_validation_samples,
# calculate_metrics,
# validate_and_save_samples,
)

# --- Hiperparâmetros da Loss Ponderada ---
white_pixel_weight = 0.6
gray_pixel_weight = 0.3
black_pixel_weight = 0.1

def main():
    # Parse arguments and load config
    parser = argparse.ArgumentParser(description="Train the flow matching model.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="/home/viplab/nicolas/AGORAVAI/MOTFM/configs/mask_conditioning.yaml",
        help="Path to the configuration file.",
    )
    args = parser.parse_args()
    config_path = args.config_path
    config = load_config(config_path)

    # Read core settings from config
    num_epochs = config["train_args"]["num_epochs"]
    num_val_samples = config["train_args"].get("num_val_samples", 5)
    batch_size = config["train_args"]["batch_size"]
    lr = config["train_args"]["lr"]
    print_every = config["train_args"].get("print_every", 1)
    val_freq = config["train_args"].get("val_freq", 5)
    root_ckpt_dir = config["train_args"]["checkpoint_dir"]
    save_samples_freq = config["train_args"].get("save_samples_freq", 50)

    # Decide which device to use
    device = (
        torch.device(config["train_args"]["device"])
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print("Using device:", device)
    
    os.makedirs(root_ckpt_dir, exist_ok=True)

    # Model configuration flags
    mask_conditioning = config["general_args"]["mask_conditioning"]
    class_conditioning = config["general_args"]["class_conditioning"]

    # Build model
    model = build_model(config["model_args"], device=device)

    # Prepare data
    data_config = config["data_args"]
    train_data = load_and_prepare_data(
        pickle_path=data_config["pickle_path"],
        split=data_config["split_train"],
        new_masking=True,
        convert_classes_to_onehot=True,
    )
    val_data = load_and_prepare_data(
        pickle_path=data_config["pickle_path"],
        split=data_config["split_val"],
        new_masking=True,
        convert_classes_to_onehot=True,
    )

    train_loader = create_dataloader(
        Images=train_data["images"],
        Masks=train_data["masks"],
        classes=train_data["classes"] if "classes" in train_data else None,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = create_dataloader(
        Images=val_data["images"],
        Masks=val_data["masks"],
        classes=val_data["classes"] if "classes" in val_data else None,
        batch_size=batch_size,
        shuffle=False,
    )

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    log_file_path = os.path.join(root_ckpt_dir, "training_log.csv")
    log_file_exists = os.path.isfile(log_file_path)
    log_file = open(log_file_path, "a")
    if not log_file_exists:
        log_file.write("epoch,train_loss,val_loss,ssim_global\n")
    
    writer = SummaryWriter(log_dir=os.path.join(root_ckpt_dir, "logs"))
    
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    # Load the latest checkpoint if available
    latest_ckpt_dir = os.path.join(root_ckpt_dir, "latest")
    start_epoch, _, best_val_score, _, _ = load_checkpoint(
        model, optimizer, checkpoint_dir=latest_ckpt_dir, device=device, valid_only=False
    )
    
    path = AffineProbPath(scheduler=CondOTScheduler())
    scaler = torch.amp.GradScaler()
    
    """ TRAINING LOOP """    
    
    print(f"Começando o treinamento da epoca {start_epoch+1}")
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        # Use tqdm for the train loader to get a per-batch progress bar
        for batch in tqdm(train_loader, desc=f"Epoca {epoch+1}/{num_epochs}", leave=False):
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(device_type="cuda"):
                im_batch = batch["images"].to(device)
                mask_batch = batch["masks"].to(device) if mask_conditioning else None
                classes_batch = batch["classes"].to(device).unsqueeze(1) if class_conditioning else None

                # Sample random initial noise, and random t
                x_0 = torch.randn_like(im_batch)
                t = torch.rand(im_batch.shape[0], device=device)
                sample_info = path.sample(t=t, x_0=x_0, x_1=im_batch)
                # Predict velocity and compute loss
                v_pred = model(
                    x=sample_info.x_t,
                    t=sample_info.t,
                    masks=mask_batch,
                    cond=classes_batch,
                )
            
            WHITE_PIXEL_VALUE = 1.0
            GRAY_PIXEL_VALUE = 0.5
            
            weight_map = torch.full_like(sample_info.dx_t, black_pixel_weight, device=device)
            if gray_pixel_weight > 0:
                weight_map[mask_batch == GRAY_PIXEL_VALUE] = gray_pixel_weight
            if white_pixel_weight > 0:
                weight_map[mask_batch == WHITE_PIXEL_VALUE] = white_pixel_weight
                
            # loss = F.mse_loss(v_pred, sample_info.dx_t)
            
            # MAE LOSS
            # loss = torch.abs(v_pred - sample_info.dx_t)
            elementwise_error = torch.abs(v_pred - sample_info.dx_t)
            weighted_error = elementwise_error * weight_map
            loss = torch.mean(weighted_error)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

        """ LOGGING & VALIDAÇÃO """
        
        avg_train_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - start_time
        print(f"[Epoca {epoch+1}/{num_epochs}] MAE Loss: {avg_train_loss:.6f}, Duração: {elapsed:.2f}s")

        if (epoch + 1) % val_freq == 0:
            model.eval()
            total_val_loss = 0.0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validating...", leave=False):
                    real_images = batch["images"].to(device)
                    masks = batch["masks"].to(device) if mask_conditioning else None
                    classes = batch["classes"].to(device).unsqueeze(1) if class_conditioning and "classes" in batch else None
                    
                    # --- A. CÁLCULO DA LOSS DE VALIDAÇÃO ---
                    t = torch.rand(real_images.shape[0], device=device)
                    x_0_noise = torch.randn_like(real_images)
                    sample_info = path.sample(t=t, x_0=x_0_noise, x_1=real_images)
                    v_pred = model(x=sample_info.x_t, t=sample_info.t, masks=masks, cond=classes)
                    target_velocity = sample_info.dx_t
                    
                    weight_map = torch.full_like(target_velocity, black_pixel_weight, device=device)
                    if gray_pixel_weight > 0: weight_map[masks == 0.5] = gray_pixel_weight
                    if white_pixel_weight > 0: weight_map[masks == 1.0] = white_pixel_weight
                    
                    loss = torch.mean(torch.abs(v_pred - target_velocity) * weight_map)
                    total_val_loss += loss.item()
                    
                    # --- B. GERAÇÃO DE IMAGENS E ATUALIZAÇÃO DA MÉTRICA SSIM ---
                    x_init = torch.randn_like(real_images)
                    solution_steps = sample_with_solver(
                        model=model, x_init=x_init, solver_config=config["solver_args"], cond=classes, masks=masks
                    )
                    generated_images = solution_steps[-1] if solution_steps.dim() == 5 else solution_steps
                    
                    generated_images_norm = normalize_tensor_to_zero_one(generated_images)
                    ssim_metric.update(generated_images_norm, real_images)

            # --- CÁLCULO FINAL E LOGGING ---
            val_loss = total_val_loss / len(val_loader)
            ssim_global = ssim_metric.compute().item()
            ssim_metric.reset() # Limpa a métrica para a próxima época

            writer.add_scalar('Loss/train', avg_train_loss, epoch + 1)
            writer.add_scalar('Loss/validation', val_loss, epoch + 1)
            writer.add_scalar('Metrics/SSIM_global', ssim_global, epoch + 1)
            
            log_line = f"{epoch+1},{avg_train_loss:.6f},{val_loss:.6f},{ssim_global:.4f}\n"
            log_file.write(log_line); log_file.flush()
            
            print(f"--- Validation Metrics (Epoch {epoch+1}) ---")
            print(f"  - Validation Loss: {val_loss:.6f}")
            print(f"  - SSIM_global: {ssim_global:.4f}")
            print("----------------------------------------")

            # --- SALVAR CHECKPOINTS E MELHOR MODELO ---
            save_checkpoint(model, optimizer, epoch + 1, config, latest_ckpt_dir, best_val_score, avg_train_loss, val_loss)
            
            current_score = ssim_global
            if current_score > best_val_score:
                print(f"New best SSIM score: {current_score:.4f}. Saving 'best' checkpoint...")
                best_val_score = current_score
                save_checkpoint(model, optimizer, epoch + 1, config, os.path.join(root_ckpt_dir, "best"), best_val_score, avg_train_loss, val_loss)

            if (epoch + 1) % save_samples_freq == 0:
                save_validation_samples(
                    model=model, val_loader=val_loader, device=device, checkpoint_dir=os.path.join(root_ckpt_dir, f"epoch_{epoch+1}"),
                    epoch=epoch + 1, solver_config=config["solver_args"], max_samples=num_val_samples,
                    class_map=train_data.get("class_map"), mask_conditioning=mask_conditioning, class_conditioning=class_conditioning
                )
            print()
                    
            
            
            # metrics = calculate_metrics(
            #     model=model, 
            #     val_loader=val_loader, 
            #     device=device,
            #     solver_config=config["solver_args"], 
            #     white_pixel_weight=white_pixel_weight,
            #     gray_pixel_weight=gray_pixel_weight,
            #     black_pixel_weight=black_pixel_weight,
            #     mask_conditioning=mask_conditioning,
            #     class_conditioning=class_conditioning,
            # )
            
            # val_loss = metrics.get("val_loss", float('inf'))
            # ssim_global = metrics.get("SSIM_global", 0.0)

            # writer.add_scalar('Loss/train', avg_train_loss, epoch + 1)
            # writer.add_scalar('Loss/validation', val_loss, epoch + 1)
            # writer.add_scalar('Metrics/SSIM_global', ssim_global, epoch + 1)
            
            # log_line = f"{epoch+1},{avg_train_loss:.6f},{val_loss:.6f},{ssim_global:.4f}\n"
            # log_file.write(log_line)
            # log_file.flush()


            # print(f"--- Validation Metrics (Epoch {epoch+1}) ---")
            # print(f"  - Validation Loss: {val_loss:.6f}")
            # print(f"  - SSIM_global: {ssim_global:.4f}")
            

            # epoch_ckpt_dir = os.path.join(root_ckpt_dir, f"epoch_{epoch+1}")
            # save_checkpoint(model, optimizer, epoch + 1, config, epoch_ckpt_dir, best_val_score, avg_train_loss, val_loss)
            # save_checkpoint(model, optimizer, epoch + 1, config, latest_ckpt_dir, best_val_score, avg_train_loss, val_loss)
            
            # current_score = ssim_global 

            # if current_score > best_val_score:
            #     print(f"Novo melhor score SSIM: {current_score:.4f} (anterior: {best_val_score:.4f}). Salvando 'best' checkpoint...")
            #     best_val_score = current_score
            #     best_ckpt_dir = os.path.join(root_ckpt_dir, "best")
            #     save_checkpoint(model, optimizer, epoch + 1, config, best_ckpt_dir, best_val_score, avg_train_loss, val_loss)
            
            # if (epoch + 1) % save_samples_freq == 0:
            #     save_validation_samples(
            #         model=model, 
            #         val_loader=val_loader, 
            #         device=device, 
            #         checkpoint_dir=epoch_ckpt_dir, 
            #         epoch=epoch + 1, 
            #         solver_config=config["solver_args"], 
            #         max_samples=num_val_samples,
            #         class_map=train_data.get("class_map"), 
            #         mask_conditioning=mask_conditioning,
            #         class_conditioning=class_conditioning,
            #     )
            
            # print() 
            
    log_file.close()
    writer.close()
    print("Training complete!")
    
    
    
    
print("NAO É A MESMA COISA LALALLALA")
        
if __name__ == "__main__":
    main()
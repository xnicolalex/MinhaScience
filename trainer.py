import argparse
import os
import sys
import time
import warnings

# Suppress most warnings for cleaner logs (comment out if debugging is needed)

warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

# Adiciona o diretório pai ao path para encontrar outros módulos

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Importações das suas funções utilitárias

from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler

from utils.general_utils import (
load_config,
load_and_prepare_data,
create_dataloader,
save_checkpoint,
load_checkpoint,
)

from utils.utils_fm import (
build_model,
#calculate_metrics,
save_validation_samples,
calculate_metrics,
# validate_and_save_samples,
)

# --- Hiperparâmetros da Loss Ponderada ---
white_pixel_weight = 0.6
gray_pixel_weight = 0.4

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

    # Load the latest checkpoint if available
    latest_ckpt_dir = os.path.join(root_ckpt_dir, "latest")
    start_epoch, _, best_val_score, _, _ = load_checkpoint(
        model, optimizer, checkpoint_dir=latest_ckpt_dir, device=device, valid_only=False
    )

    # Define path object (scheduler included)
    path = AffineProbPath(scheduler=CondOTScheduler())

    solver_config = config["solver_args"]

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        # best_val_score = 0.0
        start_time = time.time()

        # Use tqdm for the train loader to get a per-batch progress bar
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            im_batch = batch["images"].to(device)
            mask_batch = batch["masks"].to(device) if mask_conditioning else None
            classes_batch = batch["classes"].to(device).unsqueeze(1) if class_conditioning else None

            # Sample random initial noise, and random t
            x_0 = torch.randn_like(im_batch)
            t = torch.rand(im_batch.shape[0], device=device)

            # Sample the path from x_0 to x_batch
            sample_info = path.sample(t=t, x_0=x_0, x_1=im_batch)

            # Predict velocity and compute loss
            v_pred = model(
                x=sample_info.x_t,
                t=sample_info.t,
                masks=mask_batch,
                cond=classes_batch,
            )
            loss = F.mse_loss(v_pred, sample_info.dx_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Logging
        avg_loss = epoch_loss / len(train_loader)
        if (epoch + 1) % print_every == 0:
            elapsed = time.time() - start_time
            print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {avg_loss:.6f}, Time: {elapsed:.2f}s")

        # Validation & checkpoint saving
        if (epoch + 1) % val_freq == 0:
            # --- VALIDAÇÃO QUANTITATIVA ---
            metrics = calculate_metrics(
                model=model, val_loader=val_loader, device=device,
                solver_config=config["solver_args"], white_pixel_weight=white_pixel_weight,
                gray_pixel_weight=gray_pixel_weight, mask_conditioning=mask_conditioning,
                class_conditioning=class_conditioning,
            )
            
            writer = SummaryWriter()
            val_loss = metrics.get("val_loss", float('inf'))
            writer.add_scalar('Loss/validation', val_loss, epoch + 1)
            for key, value in metrics.items():
                if key != "val_loss": writer.add_scalar(f'Metrics/{key}', value, epoch + 1)
            
            print(f"--- Validation Metrics (Epoch {epoch+1}) ---")
            print(f"  - Validation Loss: {val_loss:.6f}")
            for key, value in metrics.items():
                if key != "val_loss": print(f"  - {key}: {value:.4f}")
            print("----------------------------------------")
            
            # --- VALIDAÇÃO QUALITATIVA (SALVAR AMOSTRAS) ---
            if (epoch + 1) % save_samples_freq == 0:
                epoch_ckpt_dir = os.path.join(root_ckpt_dir, f"epoch_{epoch+1}")
                save_validation_samples(
                    model=model, val_loader=val_loader, device=device, checkpoint_dir=epoch_ckpt_dir, 
                    epoch=epoch + 1, solver_config=config["solver_args"], max_samples=num_val_samples,
                    class_map=train_data.get("class_map"), mask_conditioning=mask_conditioning,
                    class_conditioning=class_conditioning,
                    # O argumento `samples_per_input` foi removido daqui.
                )
            
            # --- SALVAR CHECKPOINTS ---
            save_checkpoint(model, 
                            optimizer, 
                            epoch + 1, 
                            config, 
                            latest_ckpt_dir, 
                            best_val_score,
                            avg_loss,
                            val_loss,
                            )
            
            
            # Lógica para salvar o melhor modelo com base no score ponderado de SSIM
            w_roi = 0.7; w_contexto = 0.3
            current_score = (w_roi * metrics.get("SSIM_roi", 0)) + (w_contexto * metrics.get("SSIM_contexto", 0))

            if current_score > best_val_score:
                print(f"Novo melhor score SSIM: {current_score:.4f}. Salvando 'best' checkpoint...")
                best_val_score = current_score
                best_ckpt_dir = os.path.join(root_ckpt_dir, "best")
                save_checkpoint(model, 
                                optimizer, 
                                epoch + 1, 
                                config, 
                                best_ckpt_dir,
                                best_val_score, 
                                avg_loss, 
                                val_loss)
            
            print()

    print("Training complete!")


if __name__ == "__main__":
    main()
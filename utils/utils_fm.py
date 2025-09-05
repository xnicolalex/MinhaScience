import sys
sys.path.append("/home/viplab/nicolas/AGORAVAI/GenerativeModels")  # Add to PYTHONPATH

from GenerativeModels.generative.networks.nets.diffusion_model_unet import DiffusionModelUNet
from GenerativeModels.generative.networks.nets.controlnet import ControlNet


import os
import json
import torch
import matplotlib.pyplot as plt
import torchmetrics
from torch import nn

# from generative.networks.nets import DiffusionModelUNet, ControlNet
from flow_matching.solver import ODESolver
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath

from .general_utils import normalize_tensor_to_zero_one, save_image
from tqdm import tqdm
import numpy as np


###############################################################################
# Model Building
###############################################################################
class MergedModel(nn.Module):
    """
    Merged model that wraps a UNet and an optional ControlNet.
    Takes in x, time in [0,1], and (optionally) a ControlNet condition.
    """

    def __init__(self, unet: DiffusionModelUNet, controlnet: ControlNet = None, max_timestep=1000):
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet
        self.max_timestep = max_timestep

        # If controlnet is None, we won't do anything special in forward.
        self.has_controlnet = controlnet is not None
        self.has_conditioning = unet.with_conditioning

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor = None,
        masks: torch.Tensor = None,
    ):
        """
        Args:
            x: input image tensor [B, C, H, W].
            t: timesteps in [0,1], will be scaled to [0, max_timestep - 1].
            cond: [B,1 , conditions_dim].
            masks: [B, C, H, W] masks for conditioning.

        Returns:
            The network output (e.g. velocity, noise, or predicted epsilon).
        """
        # Scale continuous t -> discrete timesteps(If you dont want to change the embedding function in the UNet)
        t = t * (self.max_timestep - 1)
        t = t.floor().long()

        # If t is scalar, expand to batch size
        if t.dim() == 0:
            t = t.expand(x.shape[0])

        # t's shape should be [B]

        if self.has_controlnet:
            # cond is expected to be a ControlNet conditioning, e.g. mask
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                x=x, timesteps=t, controlnet_cond=masks, context=cond
            )
            output = self.unet(
                x=x,
                timesteps=t,
                context=cond,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            )
        else:
            # If no ControlNet, cond might be cross-attention or None
            output = self.unet(x=x, timesteps=t, context=cond)

        return output


def build_model(model_config: dict, device: torch.device = None) -> MergedModel:
    """
    Builds a model (UNet only, or UNet+ControlNet) based on the provided model_config.

    Args:
        model_config: Dictionary containing model configuration.
        device: Device to move the model to.

    Returns:
        A MergedModel instance.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Make a copy so the original config remains unaltered.
    mc = model_config.copy()

    # Pop out keys that are not needed by the model constructors.
    mask_conditioning = mc.pop("mask_conditioning", False)
    max_timestep = mc.pop("max_timestep", 1000)
    # Pop out ControlNet specific key, if present.
    cond_embed_channels = mc.pop("conditioning_embedding_num_channels", None)

    # Build the base UNet by passing all remaining items as kwargs.
    unet = DiffusionModelUNet(**mc)

    controlnet = None
    if mask_conditioning:
        # Ensure the controlnet has its specific key.
        if cond_embed_channels is None:
            cond_embed_channels = (16,)
        # Pass the same config kwargs to ControlNet plus the controlnet-specific key.
        controlnet = ControlNet(**mc, conditioning_embedding_num_channels=cond_embed_channels)
        controlnet.load_state_dict(unet.state_dict(), strict=False)

    model = MergedModel(unet=unet, controlnet=controlnet, max_timestep=max_timestep)

    # Print number of trainable parameters.
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params} trainable parameters.")
    model_size_mb = num_params * 4 / (1024**2)
    print(f"Model size: {model_size_mb:.2f} MB")

    return model.to(device)


def sample_with_solver(
    model,
    x_init,
    solver_config,
    cond=None,
    masks=None,
):
    """
    Uses ODESolver (flow-matching) to sample from x_init -> final output.
    solver_config might contain keys:
        {
          "method": "midpoint"/"rk4"/etc.,
          "step_size": float,
          "time_points": int,
        }

    Returns either the full trajectory [time_points, B, C, H, W] if return_intermediates=True
    or just the final state [B, C, H, W].
    """
    solver = ODESolver(velocity_model=model)

    time_points = solver_config.get("time_points", 10)
    T = torch.linspace(0, 1, time_points, device=x_init.device)

    method = solver_config.get("method", "midpoint")
    step_size = solver_config.get("step_size", 0.02)

    sol = solver.sample(
        time_grid=T,
        x_init=x_init,
        method=method,
        step_size=step_size,
        return_intermediates=True,
        cond=cond,
        masks=masks,
    )
    return sol


def plot_solver_steps(sol, im_batch, mask_batch, class_batch, class_map, outdir, max_plot=4):
    if sol.dim() != 5:  # No intermediates to plot
        return
    n_samples = min(sol.shape[1], max_plot)
    n_steps = sol.shape[0]
    if mask_batch is not None:
        fig, axes = plt.subplots(n_samples, n_steps + 2, figsize=(20, 8))
    else:
        fig, axes = plt.subplots(n_samples, n_steps + 1, figsize=(20, 8))
    if n_samples == 1:
        axes = [axes]
    for i in range(n_samples):
        for t in range(n_steps):
            axes[i][t].imshow(sol[t, i].cpu().numpy().squeeze(), cmap="gray")
            axes[i][t].axis("off")
            if i == 0:
                axes[i][t].set_title(f"Step {t}")
        col = n_steps
        if mask_batch is not None:
            axes[i][col].imshow(mask_batch[i].cpu().numpy().squeeze(), cmap="gray")
            axes[i][col].axis("off")
            if i == 0:
                axes[i][col].set_title("Mask")
            col += 1
        axes[i][col].imshow(im_batch[i].cpu().numpy().squeeze(), cmap="gray")
        axes[i][col].axis("off")
        if i == 0:
            axes[i][col].set_title("Real")
        if class_map and class_batch is not None:
            idx = class_batch[i].argmax().item()
            cls = class_map[idx] if idx < len(class_map) else str(idx)
            axes[i][col].text(
                0.5,
                -0.15,
                f"Class: {cls}",
                ha="center",
                va="top",
                transform=axes[i][col].transAxes,
                color="red",
                fontsize=9,
            )
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "solver_steps.png"), bbox_inches="tight", pad_inches=0)
    plt.close()
    
    

# # Em utils/utils_fm.py

# # ... (suas importações e outras funções)

# def calculate_metrics(
#     model: torch.nn.Module,
#     val_loader: torch.utils.data.DataLoader,
#     device: torch.device,
#     solver_config: dict,
#     white_pixel_weight: float,
#     gray_pixel_weight: float,
#     black_pixel_weight: float, # Adicionado para consistência
#     mask_conditioning: bool = True,
#     class_conditioning: bool = False,
# ):
#     """
#     Versão final: Calcula a loss de validação e a métrica SSIM Global de forma correta e eficiente.
#     """
#     # CORREÇÃO: data_range ajustado para 1.0 para corresponder à normalização [0, 1]
#     data_range = 1.0
#     ssim_global = torchmetrics.StructuralSimilarityIndexMeasure(data_range=data_range).to(device)
    
#     path = AffineProbPath(scheduler=CondOTScheduler())
#     model.eval()
#     total_val_loss = 0.0

#     try:
#         with torch.no_grad():
#             for batch in tqdm(val_loader, desc="Calculating Metrics (Loss & SSIM Global)...", leave=False):
#                 real_images = batch["images"].to(device)
#                 masks = batch["masks"].to(device) if mask_conditioning else None
#                 classes = batch["classes"].to(device).unsqueeze(1) if class_conditioning and "classes" in batch else None
                
#                 # --- A. CÁLCULO DA LOSS DE VALIDAÇÃO ---
#                 t = torch.rand(real_images.shape[0], device=device)
#                 x_0_noise = torch.randn_like(real_images)
#                 sample_info = path.sample(t=t, x_0=x_0_noise, x_1=real_images)
#                 v_pred = model(x=sample_info.x_t, t=sample_info.t, masks=masks, cond=classes)
#                 target_velocity = sample_info.dx_t
                
#                 weight_map = torch.full_like(target_velocity, black_pixel_weight, device=device)
#                 if gray_pixel_weight > 0: weight_map[masks == 0.5] = gray_pixel_weight
#                 if white_pixel_weight > 0: weight_map[masks == 1.0] = white_pixel_weight
                
#                 loss = torch.mean(torch.abs(v_pred - target_velocity) * weight_map)
#                 total_val_loss += loss.item()
                
#                 # --- B. CÁLCULO DA MÉTRICA SSIM GLOBAL ---
#                 x_init = torch.randn_like(real_images)
#                 solution_steps = sample_with_solver(
#                     model=model, x_init=x_init, solver_config=solver_config, cond=classes, masks=masks
#                 )
#                 generated_images = solution_steps[:, -1] if solution_steps.dim() == 5 else solution_steps

#                 # CORREÇÃO: A normalização foi reativada
#                 generated_images_norm = normalize_tensor_to_zero_one(generated_images)
                
#                 # CORREÇÃO: As linhas de fatiamento `[:1,:,:,:]` foram removidas
#                 ssim_global.update(generated_images_norm, real_images)

#         avg_val_loss = total_val_loss / len(val_loader)
#         metrics_results = {
#             "val_loss": avg_val_loss,
#             "SSIM_global": ssim_global.compute().item(),
#         }
#         return metrics_results
        
#     finally:
#         ssim_global.reset()
            

def validate_and_save_samples(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    checkpoint_dir: str,
    epoch: int,
    solver_config: dict,
    max_samples=16,
    class_map=None,
    mask_conditioning=True,
    class_conditioning=False,
):
    """
    Versão final, eficiente e robusta para salvar amostras de validação.
    """
    # Função auxiliar para garantir compatibilidade com JSON
    def convert_for_json(obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_for_json(x) for x in obj]
        return obj
    
    print(f"📸 Saving up to {max_samples} visual samples for epoch {epoch}...")
    
    model.eval()
    outdir = os.path.join(checkpoint_dir, "val_samples")
    os.makedirs(outdir, exist_ok=True)
    count, step_plot_done = 0, False
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Saving visual samples...", leave=False):
            if count >= max_samples:
                break
            
            imgs = batch["images"].to(device)
            cond = batch["classes"].to(device).unsqueeze(1) if class_conditioning else None
            masks = batch["masks"].to(device) if mask_conditioning else None

            x_init = torch.randn_like(imgs)
            sol = sample_with_solver(
                model, x_init, solver_config, cond=cond, masks=masks
            )
            final_imgs = sol[:, -1] if sol.dim() == 5 else sol
            
            # Limita o número de amostras a serem salvas neste batch
            num_to_save_in_batch = min(final_imgs.size(0), max_samples - count)

            for i in range(num_to_save_in_batch):
                sdir = os.path.join(outdir, f"sample_{count+1:03d}")
                os.makedirs(sdir, exist_ok=True)

                gen_img = normalize_tensor_to_zero_one(final_imgs[i])
                real_img = normalize_tensor_to_zero_one(imgs[i])
                
                save_image(gen_img, os.path.join(sdir, "generated.png"))
                save_image(real_img, os.path.join(sdir, "real.png"))
                
                if masks is not None:
                    mask_img = normalize_tensor_to_zero_one(masks[i])
                    save_image(mask_img, os.path.join(sdir, "mask.png"))
                
                if class_map and "classes" in batch:
                    idx = batch["classes"][i].argmax().item()
                    
                    metadata_dict = {
                        "class_index": idx,
                        "class_name": class_map[idx] if idx < len(class_map) else str(idx),
                        "class_map": class_map,
                        "class_conditioning": class_conditioning,
                        "mask_conditioning": mask_conditioning,
                    }
                    
                    # --- CORREÇÃO APLICADA AQUI ---
                    # Garante que todos os tipos NumPy sejam convertidos antes de salvar
                    metadata_cleaned = convert_for_json(metadata_dict)
                    
                    with open(os.path.join(sdir, "class.json"), "w") as f:
                        json.dump(metadata_cleaned, f, indent=4)
                
                count += 1
            
            if not step_plot_done:
                clz = batch["classes"] if class_map and "classes" in batch else None
                plot_solver_steps(sol, imgs, masks, clz, class_map, outdir)
                step_plot_done = True
                
                
# --- FUNÇÃO AUXILIAR PARA CORRIGIR O ERRO JSON ---
def convert_numpy_types(obj):
    """Converte recursivamente tipos NumPy para tipos nativos do Python para serialização JSON."""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(x) for x in obj]
    return obj


def save_validation_samples(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    checkpoint_dir: str,
    epoch: int,
    solver_config: dict,
    max_samples=8,
    class_map=None,
    mask_conditioning=True,
    class_conditioning=False,
):
    """
    Versão final: Salva UMA amostra gerada por imagem de entrada de forma eficiente e robusta.
    """
    print(f"📸 Saving up to {max_samples} visual samples for epoch {epoch} (1 generated per sample)...")

    model.eval()
    outdir = os.path.join(checkpoint_dir, "val_samples")
    os.makedirs(outdir, exist_ok=True)
    samples_saved = 0

    with torch.no_grad():
        for batch in val_loader:
            if samples_saved >= max_samples:
                break

            real_images = batch["images"].to(device)
            masks = batch["masks"].to(device) if mask_conditioning else None
            classes = batch["classes"].to(device).unsqueeze(1) if class_conditioning else None

            # Gera UMA imagem por entrada, de forma eficiente para o batch inteiro
            x_init = torch.randn_like(real_images)
            solution_steps = sample_with_solver(
                model=model, x_init=x_init, solver_config=solver_config, cond=classes, masks=masks
            )
            generated_images = solution_steps[:, -1] if solution_steps.dim() == 5 else solution_steps

            num_to_save_in_batch = min(real_images.size(0), max_samples - samples_saved)

            for i in range(num_to_save_in_batch):
                sdir = os.path.join(outdir, f"sample_{samples_saved+1:03d}")
                os.makedirs(sdir, exist_ok=True)
                
                # Salva a imagem real, a máscara e a ÚNICA imagem gerada
                save_image(normalize_tensor_to_zero_one(real_images[i]), os.path.join(sdir, "real.png"))
                if masks is not None:
                    save_image(normalize_tensor_to_zero_one(masks[i]), os.path.join(sdir, "mask.png"))
                save_image(normalize_tensor_to_zero_one(generated_images[i]), os.path.join(sdir, "generated.png"))

                # Salva o arquivo JSON com os metadados
                # if class_map and "classes" in batch:
                if class_map and classes is not None:
                    idx = classes[i].argmax().item()
                    
                    metadata_dict = {
                        "class_index": idx,
                        "class_name": class_map[idx] if idx < len(class_map) else str(idx),
                        "class_map": class_map,
                        "class_conditioning": class_conditioning,
                        "mask_conditioning": mask_conditioning,
                    }
                    
                    # Usa a função auxiliar para "limpar" o dicionário ANTES de salvar
                    metadata_cleaned = convert_numpy_types(metadata_dict)
                    
                    with open(os.path.join(sdir, "class.json"), "w") as f:
                        json.dump(metadata_cleaned, f, indent=4)
                
                samples_saved += 1

    print(f"✅ {samples_saved} amostras de validação salvas em: {outdir}")

# def validate_and_save_samples(
#     model: torch.nn.Module,
#     val_loader: torch.utils.data.DataLoader,
#     device: torch.device,
#     checkpoint_dir: str,
#     epoch: int,
#     solver_config: dict,
#     max_samples=16,
#     class_map=None,
#     mask_conditioning=True,
#     class_conditioning=False,
# ):
    
#     def convert_for_json(obj):
#         if isinstance(obj, (np.integer, np.int32, np.int64)):
#             return int(obj)
#         elif isinstance(obj, (np.floating, np.float32, np.float64)):
#             return float(obj)
#         elif isinstance(obj, np.ndarray):
#             return obj.tolist()
#         elif isinstance(obj, dict):
#             return {k: convert_for_json(v) for k, v in obj.items()}
#         elif isinstance(obj, (list, tuple)):
#             return [convert_for_json(x) for x in obj]
#         return obj
    
#     model.eval()
#     outdir = os.path.join(checkpoint_dir, f"val_samples_epoch_{epoch}")
#     os.makedirs(outdir, exist_ok=True)
#     count, step_plot_done = 0, False
    
#     for batch in tqdm(val_loader, desc="Validating"):
#         imgs = batch["images"].to(device)
#         cond = batch["classes"].to(device).unsqueeze(1) if class_conditioning else None
#         masks = batch["masks"].to(device) if mask_conditioning else None

#         x_init = torch.randn_like(imgs)
#         sol = sample_with_solver(
#             model,
#             x_init,
#             solver_config,
#             cond=cond,
#             masks=masks,
#         )
#         final_imgs = sol[-1] if sol.dim() == 5 else sol
#         for i in range(final_imgs.size(0)):
#             if count >= max_samples:
#                 break
#             gen_img = normalize_zero_to_one(final_imgs[i])
#             real_img = normalize_zero_to_one(imgs[i])
#             sdir = os.path.join(outdir, f"sample_{count+1:03d}")
#             os.makedirs(sdir, exist_ok=True)
#             save_image(gen_img, os.path.join(sdir, "gen.png"))
#             save_image(real_img, os.path.join(sdir, "real.png"))
#             if masks is not None:
#                 cnd_img = normalize_zero_to_one(masks[i])
#                 save_image(cnd_img, os.path.join(sdir, "mask.png"))
#             if class_map and "classes" in batch:
#                 idx = batch["classes"][i].argmax().item()
#                 with open(os.path.join(sdir, "class.json"), "w") as f:
#                     json.dump(
#                         {
#                             "class_index": idx,
#                             "class_name": class_map[idx] if idx < len(class_map) else str(idx),
#                             "class_map": class_map,
#                             "class_coditioning": class_conditioning,
#                             "mask_conditioning": mask_conditioning,
#                         },
#                         f,
#                         indent=4,
#                     )
#             count += 1
#         if not step_plot_done:
#             clz = batch["classes"] if class_map and "classes" in batch else None
#             plot_solver_steps(sol, imgs, masks, clz, class_map, outdir)
#             step_plot_done = True
#         if count >= max_samples:
#             break
#     print(f"Validation samples saved in: {outdir}")


@torch.no_grad()
def sample_batch(
    model: torch.nn.Module,
    solver_config: dict,
    batch: torch.Tensor,
    device: torch.device,
    class_conditioning: bool = False,
    mask_conditioning: bool = False,
):
    model.eval()
    imgs = batch["images"].to(device)
    cond = batch["classes"].to(device).unsqueeze(1) if class_conditioning else None
    masks = batch["masks"].to(device) if mask_conditioning else None

    x_init = torch.randn_like(imgs)
    sol = sample_with_solver(
        model=model, solver_config=solver_config, x_init=x_init, cond=cond, masks=masks
    )
    final_imgs = sol[-1] if sol.dim() == 5 else sol
    return final_imgs


print("Não é a mesma coisa lalala")
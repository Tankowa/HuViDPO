import argparse
import datetime
from itertools import combinations
import logging
import inspect
import math
import os
import random
import cv2
from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from peft import LoraConfig, get_peft_model
import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from HuViDPO.data.dataset_DPO import DPOdatasets
from HuViDPO.models.unet import UNet3DConditionModel
from HuViDPO.data.dataset import HuViDPODataset
from HuViDPO.pipelines.pipeline_HuViDPO import HuViDPO
from HuViDPO.util import save_videos_grid, ddim_inversion
from einops import rearrange
import json
import copy
import numpy as np
# import os
# os.environ['LD_LIBRARY_PATH'] = '/data/anaconda3/'
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")

loss_file = open("loss_DPO.txt", "a")

def his_match(src, dst):
    src = src * 255.0
    dst = dst * 255.0
    src = src.astype(np.uint8)
    dst = dst.astype(np.uint8)
    res = np.zeros_like(dst)

    cdf_src = np.zeros((3, 256))
    cdf_dst = np.zeros((3, 256))
    cdf_res = np.zeros((3, 256))
    kw = dict(bins=256, range=(0, 256), density=True)
    for ch in range(3):
        his_src, _ = np.histogram(src[:, :, ch], **kw)
        hist_dst, _ = np.histogram(dst[:, :, ch], **kw)
        cdf_src[ch] = np.cumsum(his_src)
        cdf_dst[ch] = np.cumsum(hist_dst)
        index = np.searchsorted(cdf_src[ch], cdf_dst[ch], side='left')
        np.clip(index, 0, 255, out=index)
        res[:, :, ch] = index[dst[:, :, ch]]
        his_res, _ = np.histogram(res[:, :, ch], **kw)
        cdf_res[ch] = np.cumsum(his_res)
    return res / 255.0

def shuffle_samples(samples):
    random.shuffle(samples)
    return samples

def load_sample(json_file_path):
    with open(json_file_path, 'r') as f:
        sample_data = json.load(f)

    processed_data = []
    for sample in sample_data:
        processed_sample = {}
        for key, value in sample.items():
            if key in ["prompt_embeds", "timesteps", "latents", "next_latents"]:
                processed_sample[key] = torch.tensor(value).to('cuda')  # 将张量移动到指定设备
            elif key == "score":
                processed_sample[key] = torch.tensor([float(value)]).to('cuda')
            elif key == "video":
                processed_sample[key] = value  # 字符串不需要移动设备
        processed_data.append(processed_sample)

    return processed_data

def main(
    pretrained_model_path: str,
    output_dir: str,
    train_data: Dict,
    validation_data: Dict,
    validation_steps: int = 100,
    trainable_modules: Tuple[str] = (
        "attn1.to_q",
        "attn2.to_q",
        "attn_temp",
    ),
    train_batch_size: int = 4,
    max_train_steps: int = 500,
    learning_rate: float = 3e-5,
    scale_lr: bool = False,
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 0,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = True,
    checkpointing_steps: int = 500,
    resume_from_checkpoint: Optional[str] = None,
    mixed_precision: Optional[str] = "fp16",
    use_8bit_adam: bool = False,
    enable_xformers_memory_efficient_attention: bool = True,
    seed: Optional[int] = None,
):
    *_, config = inspect.getargvalues(inspect.currentframe())

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if seed is not None:
        set_seed(seed)

    # Handle the output folder creation
    if accelerator.is_main_process:
        # now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        # output_dir = os.path.join(output_dir, now)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/inv_latents", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16


    # 设置 LoRA 配置，确保 lora_dropout 被初始化
    lora_config = LoraConfig(
        r=16,  # 低秩适配器的秩
        lora_alpha=32,  # LoRA 的 alpha 超参数
        init_lora_weights="gaussian",  # 初始化权重方式
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],  # 目标模块名称
        lora_dropout=0.1  # 添加此行以设置 dropout
    )
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")

    my_model_path = args.weights
    unet = UNet3DConditionModel.from_pretrained('/'.join(my_model_path.split('/')[:-1]), subfolder=my_model_path.split('/')[-1]).to('cuda').to(weight_dtype)
    #unet = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet")
    
    unet.requires_grad_(False)

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    """
    for name, module in unet.named_modules():
        if name.endswith(tuple(trainable_modules)):
            for params in module.parameters():
                params.requires_grad = True    
    
    """

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()


    ref = copy.deepcopy(unet)
    for params in ref.parameters():
        params.requires_grad = False


        # 将 LoRA 适配器添加到模型中
    unet = get_peft_model(unet, lora_config)


    for name, module in unet.named_modules():
        if "lora" in name.lower():  # 或者你可以使用具体的模块名
            print(f"Found LoRA adapter in: {name}, Module: {module}")
    

    for name, param in unet.named_parameters():
        if param.requires_grad:
            param.data = param.to(torch.float32)
            print(f"Converted {name} to float32")

    if scale_lr:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=1e-4,  # 调整学习率到更合理的范围
        betas=(adam_beta1, adam_beta2),
        weight_decay=1e-4,  # 可以稍微降低 weight_decay 以避免过强的正则化
        eps=1e-3  # 设置较小的 eps 来避免梯度更新不稳定
    )


    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)


    logger.info("***** Running training *****")
    global_step = 0
    first_epoch = 0
    num_train_epochs = 12


    # 假设你只有一个 JSON 文件，路径如下
    json_file_path = train_data.json_file_path
    #json_file_path = "data/video/scores1.json"

    # 加载 samples
    samples = load_sample(json_file_path)

    #print("samples",samples)

    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        train_loss = 0.0

        total_samples = len(samples)  # 这里应该是 samples 的长度，而不是 num_videos * NUM_PER_PROMPT
        #print("total_samples",total_samples)
        # 在每个 epoch 中随机打乱样本
        shuffled_samples = shuffle_samples(samples)
        
        # 每次随机组合样本，生成组合对
        sample_combinations = list(combinations(range(total_samples), 2))
        random.shuffle(sample_combinations)  # 随机打乱组合顺序

        #print("sample_combinations",sample_combinations)
        w=0
        
        for comb in sample_combinations:
            w=w+1
            idx1, idx2 = comb

            shuffled_samples[idx1]['score']
            
            if shuffled_samples[idx1]['score'] > shuffled_samples[idx2]['score']:
                video_name_1 = shuffled_samples[idx1]['video']
                video_name_2 = shuffled_samples[idx2]['video']
                score_1 = shuffled_samples[idx1]['score']
                score_2 = shuffled_samples[idx2]['score']
            else:
                video_name_1 = shuffled_samples[idx2]['video']
                video_name_2 = shuffled_samples[idx1]['video']
                score_1 = shuffled_samples[idx2]['score']
                score_2 = shuffled_samples[idx1]['score']

            # Get the training dataset
            train_dataset = DPOdatasets(video_root_1=video_name_1,video_root_2=video_name_2,score_1=score_1,score_2=score_2,path=train_data.video_root)


            # 获取视频数据
            video_data = train_dataset.get_videos()

            video_1_pixel_values = video_data["video_1_pixel_values"].to(weight_dtype).to('cuda')  # 这可能是一个包含多个样本的张量
            video_2_pixel_values = video_data["video_2_pixel_values"].to(weight_dtype).to('cuda')  # 这可能是一个包含多个样本的张量
            prompt_ids_1 = tokenizer(config['train_data']['prompt'],max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids[0].to('cuda')
            prompt_ids_2 = tokenizer(config['train_data']['prompt'],max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids[0].to('cuda')

            video_1_pixel_values = video_1_pixel_values.unsqueeze(0)
            video_2_pixel_values = video_2_pixel_values.unsqueeze(0)
            prompt_ids_1 = prompt_ids_1.unsqueeze(0)
            prompt_ids_2 = prompt_ids_2.unsqueeze(0)

            video_length = video_1_pixel_values.shape[1]
            video_1_pixel_values = rearrange(video_1_pixel_values, "b f c h w -> (b f) c h w")

            #print("video_1_pixel_values",video_1_pixel_values.size())
            latents_1 = vae.encode(video_1_pixel_values).latent_dist.sample()
            latents_1 = rearrange(latents_1, "(b f) c h w -> b c f h w", f=video_length)
            latents_1 = latents_1 * 0.18215

            video_2_pixel_values = rearrange(video_2_pixel_values, "b f c h w -> (b f) c h w")
            latents_2 = vae.encode(video_2_pixel_values).latent_dist.sample()
            latents_2 = rearrange(latents_2, "(b f) c h w -> b c f h w", f=video_length)
            latents_2 = latents_2 * 0.18215

            noise = torch.randn_like(latents_1) 
            # Sample a random timestep for each image
            bsz = latents_1.shape[0]
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents_1.device)
            
            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents_1 = noise_scheduler.add_noise(latents_1, noise, timesteps)
            noisy_latents_2 = noise_scheduler.add_noise(latents_2, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states_1 = text_encoder(prompt_ids_1)[0]
            encoder_hidden_states_2 = text_encoder(prompt_ids_2)[0]
        
            target = noise
            noisy_latents_1[:, :, 0:1, :, :] = latents_1[:, :, 0:1, :, :]
            noisy_latents_2[:, :, 0:1, :, :] = latents_2[:, :, 0:1, :, :]

            # Predict the noise residual and compute loss
            model_pred_1 = unet(noisy_latents_1, timesteps, encoder_hidden_states_1).sample
            model_losses_w = F.mse_loss(model_pred_1[:, :, 1:, :, :].float(), target[:, :, 1:, :, :].float(), reduction="mean")
            model_pred_2 = unet(noisy_latents_2, timesteps, encoder_hidden_states_2).sample
            model_losses_l = F.mse_loss(model_pred_2[:, :, 1:, :, :].float(), target[:, :, 1:, :, :].float(), reduction="mean")

            model_diff = model_losses_w - model_losses_l

            # Predict the noise residual and compute loss
            ref_pred_1 = ref(noisy_latents_1, timesteps, encoder_hidden_states_1).sample
            ref_losses_w = F.mse_loss(ref_pred_1[:, :, 1:, :, :].float(), target[:, :, 1:, :, :].float(), reduction="mean")
            ref_pred_2 = ref(noisy_latents_2, timesteps, encoder_hidden_states_2).sample
            ref_losses_l = F.mse_loss(ref_pred_2[:, :, 1:, :, :].float(), target[:, :, 1:, :, :].float(), reduction="mean")

            ref_diff = ref_losses_w - ref_losses_l

            scale_term = -0.5 * args.beta_dpo
            inside_term = scale_term * (model_diff - ref_diff)

            loss=-1.0 * F.logsigmoid(inside_term).mean()
            print("loss",loss, 10 * model_losses_w,10 * model_losses_l)
            loss_file.write(f"Loss at epoch {epoch}, step {w}: {loss.item()}\n")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
            #accelerator.backward(loss)
            #if accelerator.sync_gradients:
            #    accelerator.clip_grad_norm_(unet.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()   
        
        # epoch 结束时，可以选择保存模型或打印损失
        print(f"Epoch {epoch + 1}/{num_train_epochs} completed.")

        # 每 3 个 epoch 保存一次模型
        if (epoch + 1) % 1 == 0:
            #model_save_path = f"/data/jianglifan/model_weights_epoch_{epoch + 1}.pth"

            model_save_path = os.path.join(validation_data.save_path, f"model_weights_epoch_{epoch + 1}.pth")
            torch.save(unet.state_dict(), model_save_path)
            print(f"Model weights saved at {model_save_path}")
            
            #model_save_path = f"{model_dir}/diffusion_pytorch_model_epoch_{epoch + 1}.bin"
            #torch.save(unet.state_dict(), model_save_path)
            #print(f"Model weights saved at {model_save_path}")

            generator = torch.Generator(device='cuda')


            for idx, prompt in enumerate(validation_data.prompts):
                image = cv2.imread(os.path.join(validation_data.image_path, prompt.replace(' ', '_') + '.png'))
                #image = cv2.imread(img_path)
                image = cv2.resize(image, (512, 320))[:, :, ::-1]
                first_frame_latents = torch.Tensor(image.copy()).to('cuda').type(torch.float16).permute(2, 0, 1).repeat(1, 1, 1, 1)
                first_frame_latents = first_frame_latents / 127.5 - 1.0
                first_frame_latents = vae.encode(first_frame_latents).latent_dist.sample() * 0.18215
                first_frame_latents = first_frame_latents.repeat(1, 1, 1, 1, 1).permute(1, 2, 0, 3, 4)

                # video generation
                #video = pipe(prompt, generator=generator, latents=first_frame_latents, video_length=args.length, height=args.height, width=args.width, num_inference_steps=50, guidance_scale=args.cfg, use_inv_latent=False, num_inv_steps=50, ddim_inv_latent=ddim_inv_latent)['videos']

                pipe = HuViDPO(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")).to("cuda")
                pipe.enable_vae_slicing()
                video = pipe(prompt, generator=generator, latents=first_frame_latents, video_length=16, height=320, width=512, num_inference_steps=50, guidance_scale=12.5, use_inv_latent=False, num_inv_steps=50, ddim_inv_latent=None)['videos']

                for f in range(1, video.shape[2]):
                    former_frame = video[0, :, 0, :, :].permute(1, 2, 0).cpu().numpy()
                    frame = video[0, :, f, :, :].permute(1, 2, 0).cpu().numpy()
                    result = his_match(former_frame, frame)
                    result = torch.Tensor(result).type_as(video).to(video.device)
                    video[0, :, f, :, :] = result.permute(2, 0, 1)
                
                save_path = validation_data.save_path

                #save_path = args.output

                save_path = os.path.join(save_path, f"epoch_{epoch}_idx_{idx}.gif")

                save_videos_grid(video, save_path)


    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        pipeline = HuViDPO.from_pretrained(
            pretrained_model_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
        )
        pipeline.save_pretrained(output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/tuneavideo.yaml")
    parser.add_argument('--weights', type=str, default=None, help='Path for of model weights')
    parser.add_argument(
        "--beta_dpo",
        type=int,
        default=50,
        help="DPO KL Divergence penalty.",
    )
    args = parser.parse_args()
    import os
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"

    # 你的代码

    main(**OmegaConf.load(args.config))
from HuViDPO.pipelines.pipeline_HuViDPO import HuViDPO
from HuViDPO.models.unet import UNet3DConditionModel
from HuViDPO.util import save_videos_grid
import torch
import cv2
import random
import numpy as np
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate.utils import set_seed
import os
import imageio
from einops import rearrange
import argparse
from peft import LoraConfig, get_peft_model

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default=None, help='Path for of model weights')
    parser.add_argument('--pretrain_weight', type=str, default='./checkpoints/stable-diffusion-v1-4', help='Path for pretrained weight (SD v1.4)')
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument('--lora_weights', type=str, default=None, help='Output folder')
    parser.add_argument('--image_path', type=str, default=None, help='Output folder')
    parser.add_argument('--output_path', type=str, default=None, help='Output folder')
    parser.add_argument('--prompt_path', type=str, default=None, help='Output folder')
    parser.add_argument('--first_frame_path', type=str, default=None, help='The path for first frame image')
    parser.add_argument('-p', '--prompt', type=str, default=None, help='The video prompt. Default value: same to the filename of the first frame image')
    parser.add_argument('-hs', '--height', type=int, default=320, help='video height')
    parser.add_argument('-ws', '--width', type=int, default=512, help='video width')
    parser.add_argument('-l', '--length', type=int, default=16, help='video length')
    parser.add_argument('--cfg', type=float, default=12.5, help='classifier-free guidance scale')
    parser.add_argument('--editing', action="store_true", help='video editing')
    args = parser.parse_args()

    # load weights
    pretrained_model_path = args.pretrain_weight
    my_model_path = args.weight
    unet = UNet3DConditionModel.from_pretrained('/'.join(my_model_path.split('/')[:-1]), subfolder=my_model_path.split('/')[-1], torch_dtype=torch.float16).to('cuda')
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer", torch_dtype=torch.float16)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder", torch_dtype=torch.float16).to('cuda')
    print(args.editing)
    if args.editing:
        ddim_inv_latent = torch.load(f"{'/'.join(my_model_path.split('/')[:-1])}/inv_latents/ddim_latent-500.pt").to(torch.float16)
    else:
        ddim_inv_latent = None

    # build pipeline
    unet.enable_xformers_memory_efficient_attention()
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae", torch_dtype=torch.float16).to('cuda')
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    pipe = HuViDPO(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")).to("cuda")
    pipe.enable_vae_slicing()
    generator = torch.Generator(device='cuda')

    
    # 设置 LoRA 配置，确保 lora_dropout 被初始化
    lora_config = LoraConfig(
        r=16,  # 低秩适配器的秩
        lora_alpha=32,  # LoRA 的 alpha 超参数
        init_lora_weights="gaussian",  # 初始化权重方式
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],  # 目标模块名称
        lora_dropout=0.1  # 添加此行以设置 dropout
    )

    if args.lora_weights:
        # 将 LoRA 适配器添加到模型中
        unet = get_peft_model(unet, lora_config)

        #lora_weights_path = '/data/jianglifan/outputs/firework/model_weights_epoch_9.pth'
        lora_weights_path = args.lora_weights
        unet.load_state_dict(torch.load(lora_weights_path), strict=False)



    for name, module in unet.named_modules():
        if "lora" in name.lower():  # 或者你可以使用具体的模块名
            print(f"Found LoRA adapter in: {name}, Module: {module}")
    

    for name, param in unet.named_parameters():
        if param.requires_grad:
            param.data = param.to(torch.float32)
            print(f"Converted {name} to float32")    

    # 定义要处理的文件夹和文件路径
    image_folder = args.image_path
    prompt_file = args.prompt_path
    output_folder = args.output_path


    # 检查输出文件夹是否存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 读取prompt文件中的每一行
    with open(prompt_file, 'r') as f:
        prompts = f.readlines()

    # 遍历prompts，按顺序处理每个prompt
    for i, prompt in enumerate(prompts):
        prompt = prompt.strip()  # 移除前后的空格或换行符

        # 定义对应的图片路径，假设图片命名为1.png, 2.png, ...
        img_path = os.path.join(image_folder, f"{i + 1}.png")

        # 读取并处理图片
        print(f"Processing prompt: {prompt}, with image: {img_path}")
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"Image {img_path} not found, skipping.")
            continue

        image = cv2.resize(image, (512, 320))[:, :, ::-1]
        first_frame_latents = torch.Tensor(image.copy()).to('cuda').type(torch.float16).permute(2, 0, 1).repeat(1, 1, 1, 1)
        first_frame_latents = first_frame_latents / 127.5 - 1.0
        first_frame_latents = vae.encode(first_frame_latents).latent_dist.sample() * 0.18215
        first_frame_latents = first_frame_latents.repeat(1, 1, 1, 1, 1).permute(1, 2, 0, 3, 4)

        # 视频生成
        video = pipe(prompt, generator=generator, latents=first_frame_latents, video_length=args.length, height=args.height, width=args.width, num_inference_steps=50, guidance_scale=args.cfg, use_inv_latent=False, num_inv_steps=50, ddim_inv_latent=ddim_inv_latent)['videos']

        # 对生成的视频帧进行颜色匹配
        for f in range(1, video.shape[2]):
            former_frame = video[0, :, 0, :, :].permute(1, 2, 0).cpu().numpy()
            frame = video[0, :, f, :, :].permute(1, 2, 0).cpu().numpy()
            result = his_match(former_frame, frame)
            result = torch.Tensor(result).type_as(video).to(video.device)
            video[0, :, f, :, :] = result.permute(2, 0, 1)

        # 定义输出文件路径
        save_path = os.path.join(output_folder, f"{i + 1}.gif")

        # 保存视频
        save_videos_grid(video, save_path)

        print(f"Saved video for prompt {i+1} at {save_path}")

if __name__ == '__main__':
    main()
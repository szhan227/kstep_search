import torch
import numpy as np
from diffusers import StableVideoDiffusionPipeline, CogVideoXImageToVideoPipeline
from diffusers import AnimateDiffPipeline
from diffusers.utils import load_image, export_to_video, export_to_gif
from diffusers import DEISMultistepScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg
from PIL import Image
from torchvision import transforms
from transformers import AutoProcessor, CLIPVisionModelWithProjection
from einops import rearrange
from opensora.sample.pipeline_iterative import OpenSoraIterativePipeline

# from dataset import WebVideoDataset
from tqdm import tqdm
import os
import torchvision
import glob
from PIL import Image

from decord import VideoReader

import glob

import argparse
import random

from omegaconf import OmegaConf


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pipe", type=str, default="SVD", help="Choose from SVD, OPEN_SORA_PLAN, COGVIDEOX")
    parser.add_argument("--video_save_dir", type=str, default="./outputs")
    parser.add_argument("--conditional_image_path", type=str, required=True)
    parser.add_argument("--num_chunks", type=int, default=5)
    parser.add_argument("--num_candidates", type=int, default=5)
    parser.add_argument("--num_sampling_steps", type=int, default=50)
    parser.add_argument("--kstep", type=int, default=3)
    parser.add_argument("--fps", type=int, default=8)

    args = parser.parse_args()

    return args


def get_pipeline(name, device, config=None):

    if name == "SVD":
        pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
        ).to(device)

    elif name == "COGVIDEOX":
        pipe = CogVideoXImageToVideoPipeline.from_pretrained("THUDM/CogVideoX-5b-I2V", torch_dtype=torch.float16).to(device)

    elif name == "OPEN_SORA_PLAN":


        print("Start to import OpenSoraPlan...")
        from diffusers.schedulers import EulerAncestralDiscreteScheduler
        from opensora.models.causalvideovae import ae_stride_config, ae_wrapper
        from opensora.sample.pipeline_iterative import OpenSoraIterativePipeline
        from opensora.models.diffusion.opensora_v1_3.modeling_inpaint import OpenSoraInpaint_v1_3
        from transformers import AutoTokenizer, MT5EncoderModel, CLIPTextModelWithProjection

        cache_dir = "./cache_dir"
        dtype = torch.float16

        weight_dtype = dtype
        print("Start to load VAE...")
        vae = ae_wrapper[config.ae](config.ae_path)
        vae.vae = vae.vae.to(device=device, dtype=weight_dtype).eval()
        vae.vae_scale_factor = ae_stride_config[config.ae]
        vae.vae.enable_tiling()

        print("Start to load TextEncoder...")
        text_encoder_1 = MT5EncoderModel.from_pretrained(
            config.text_encoder_name_1, cache_dir=config.cache_dir, 
            torch_dtype=weight_dtype
        ).eval()
        
        tokenizer_1 = AutoTokenizer.from_pretrained(
            config.text_encoder_name_1, cache_dir=config.cache_dir
        )

        print("Start to load DiT...")
        transformer_model = OpenSoraInpaint_v1_3.from_pretrained(
            config.model_path, cache_dir=config.cache_dir,
            device_map=None, torch_dtype=weight_dtype
            ).eval()
        

        scheduler_kwargs = dict(
            prediction_type=config.prediction_type, 
            rescale_betas_zero_snr=config.rescale_betas_zero_snr, 
            timestep_spacing="trailing" if config.rescale_betas_zero_snr else 'leading', 
        )

        scheduler = EulerAncestralDiscreteScheduler(scheduler_kwargs)

        print("Start to load OpenSoraPlanIterativePipeline...")

        pipe = OpenSoraIterativePipeline(
            vae=vae,
            text_encoder=text_encoder_1,
            tokenizer=tokenizer_1,
            scheduler=scheduler,
            transformer=transformer_model, 
        ).to(device)

    else:
        # Customize your own pipeline here
        raise NotImplementedError("Pipeline is not supported")
    
    return pipe

def get_config(name):

    if name == "SVD":
        config = OmegaConf.load("./configs/svd_inference.yaml")
    elif name == "COGVIDEOX":
        config = OmegaConf.load("./configs/cogvideox_inference.yaml")
    elif name == "OPEN_SORA_PLAN":
        config = OmegaConf.load("./configs/opensoraplan_inference.yaml")
    else:
        # Customize your own config here
        raise NotImplementedError("Config is not supported")
    return config


def pipeline_forward(pipe,
                     prompt,
                     conditional_image,
                     num_sampling_steps,
                     num_frames,
                     height,
                     width,
                     latents=None,
                     decode_chunk_size=8):
    
    
    if isinstance(pipe, StableVideoDiffusionPipeline):
        frames = pipe(
            conditional_image, 
            num_frames=num_frames,
            num_sampling_steps=num_sampling_steps,
            height=height,
            width=width,
            decode_chunk_size=decode_chunk_size,
            latents=latents,
        ).frames[0]

        # list of PIL.Image
        return frames
    
    elif isinstance(pipe, CogVideoXImageToVideoPipeline):
        frames = pipe(
            prompt=prompt,
            height=height,
            width=width,
            image=conditional_image,  # The path of the image to be used as the background of the video
            num_videos_per_prompt=1,  # Number of videos to generate per prompt
            num_inference_steps=num_sampling_steps,  # Number of inference steps
            num_frames=num_frames,  # Number of frames to generate，changed to 49 for diffusers version `0.30.3` and after.
            use_dynamic_cfg=True,  # This id used for DPM Sechduler, for DDIM scheduler, it should be False
            guidance_scale=6.0,
            generator=torch.Generator().manual_seed(42),  # Set the seed for reproducibility
            latents=latents,
        ).frames[0]  # list of PIL image

        return frames
    
    elif isinstance(pipe, OpenSoraIterativePipeline):

        positive_prompt = """
        high quality, high aesthetic, {}
        """

        negative_prompt = """
        nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, 
        low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry.
        """

        input_prompt = positive_prompt.format(prompt)

        frames = pipe(
            conditional_pixel_values_path=None,
            conditional_pixel_values_input=conditional_image,
            mask_type=None,
            crop_for_hw=False,
            max_hxw=236544,
            # noise_strength=args.noise_strength,
            prompt=input_prompt, 
            negative_prompt=negative_prompt, 
            num_frames=args.num_frames,
            height=height,
            width=width,
            num_inference_steps=num_sampling_steps,
            guidance_scale=7.5,
            num_samples_per_prompt=args.num_samples_per_prompt,
            max_sequence_length=args.max_sequence_length,
            latents=latents,
            seed_for_step=42, # to make sure same noise input -> same output
        )

        frames = frames['videos'][0]
        # List of PIL.Image
        frames = [Image.fromarray(frame.cpu().numpy()) for frame in frames]
        return frames
        


def postprocess_video_chunk(frames):
    # to make frames to be a list of PIL images

    if len(frames) < 1:
        raise ValueError("Frames should not be empty")
    
    if isinstance(frames, list) and isinstance(frames[0], Image.Image):
        return frames
    elif isinstance(frames, torch.Tensor):
        # reshape tp t h w c
        if frames.shape[1] == 3:
            frames = rearrange(frames, "t c h w -> t h w c")
        
        # range to be uint8
        if frames.dtype != torch.uint8:
            frames = (frames * 255).clamp(0, 255).to(torch.uint8)
        
        # to PIL images
        frames = [Image.fromarray(frame.numpy()) for frame in frames]
        return frames
    
    else:
        raise ValueError("Frames should be either list of PIL images or torch.Tensor")
    

def k_step_sampling(args):

    device = "cuda"

    pipeline_name = "SVD"

    config = get_config(pipeline_name)

    pipe = get_pipeline(pipeline_name, device, config)

    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_model.eval()

    conditional_image = Image.open(args.conditional_image_path).convert("RGB")

    video_chunks = list()

    prompt = "A beautiful waterfall in the forest."

    for i in range(args.num_chunks):

        init_noises = [torch.randn((1, config.num_frames, config.latent_c, config.latent_h, config.latent_w), dtype=torch.float16).to(device) for _ in range(5)]
        candidate_videos = list()

        for j, noise in enumerate(init_noises):

            
            # list of PIL.Image
            candidate_video = pipeline_forward(pipe=pipe, 
                                               prompt=prompt,
                                               conditional_image=conditional_image, 
                                               num_sampling_steps=args.kstep, 
                                               num_frames=config.num_frames, 
                                               height=config.height, 
                                               width=config.width, 
                                               latents=noise, 
                                               decode_chunk_size=8)
            
            candidate_video = torch.stack([torch.from_numpy(np.array(frame)) for frame in candidate_video]) # t h w c, torch.uint8

            candidate_videos.append(candidate_video)

        cand_v_embeds = []
        for cand_video in candidate_videos:
            processed_video = processor(images=cand_video, return_tensors="pt")['pixel_values'].to(device)
            video_embed = clip_model(processed_video).image_embeds
            cand_v_embeds.append(video_embed)
        
        processed_cond_image = processor(images=conditional_image, return_tensors="pt")['pixel_values'].to(device)
        cond_image_embed = clip_model(processed_cond_image).image_embeds # [1, 512]
        cond_image_embed = cond_image_embed.repeat(config.num_frames, 1) # [num_frames, 512]

        similarities = [torch.nn.functional.cosine_similarity(cond_image_embed, cand_v_embed).min().item() for cand_v_embed in cand_v_embeds]
        similarities = torch.tensor(similarities)

        best_noise_index = similarities.argmax().item()

        # Use best noise picked
        best_noise = init_noises[best_noise_index].detach().clone()

        frames = pipeline_forward(
            pipe=pipe, 
            prompt=prompt,
            conditional_image=conditional_image, 
            num_sampling_steps=args.num_sampling_steps, 
            num_frames=config.num_frames, 
            height=config.height, 
            width=config.width, 
            latents=best_noise, 
            decode_chunk_size=8)

        frames = postprocess_video_chunk(frames)

        conditional_image = frames[-1]

        current_chunk = torch.from_numpy(np.array(frames)) # t h w c

        torchvision.io.write_video(os.path.join(args.video_save_path, f"video_chunk_{i}.mp4"), current_chunk, fps=args.fps)

        video_chunks.append(current_chunk)

    long_video = torch.cat(video_chunks, dim=0)

    torchvision.io.write_video(os.path.join(args.video_save_path, "long_video.mp4"), long_video, fps=args.fps)


if __name__ == "__main__":
    args = parse_args()
    k_step_sampling(args)

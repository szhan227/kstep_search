print("Start to call sample_utils.py")
from diffusers.schedulers import (
    DDIMScheduler, DDPMScheduler, PNDMScheduler,
    EulerDiscreteScheduler, DPMSolverMultistepScheduler,
    HeunDiscreteScheduler, EulerAncestralDiscreteScheduler,
    DEISMultistepScheduler, KDPM2AncestralDiscreteScheduler, 
    DPMSolverSinglestepScheduler, CogVideoXDDIMScheduler, 
    FlowMatchEulerDiscreteScheduler
    )
from einops import rearrange
import time
import torch
import os
import torch.distributed as dist
from torchvision.utils import save_image
import imageio
import math
import argparse
from transformers import AutoModelForCausalLM
import torchvision
from PIL import Image
import numpy as np
import random
from transformers import AutoProcessor, CLIPVisionModelWithProjection


try:
    import torch_npu
    from opensora.npu_config import npu_config
    from opensora.acceleration.parallel_states import initialize_sequence_parallel_state, hccl_info
except:
    torch_npu = None
    npu_config = None
    from opensora.utils.parallel_states import initialize_sequence_parallel_state, nccl_info
    pass

print("Start to import OpenSoraPlan...")
from opensora.utils.utils import set_seed
from opensora.models.causalvideovae import ae_stride_config, ae_wrapper
from opensora.sample.pipeline_opensora import OpenSoraPipeline
from opensora.sample.pipeline_inpaint import OpenSoraInpaintPipeline
from opensora.sample.pipeline_iterative import OpenSoraIterativePipeline
from opensora.models.diffusion.opensora_v1_3.modeling_opensora import OpenSoraT2V_v1_3
from opensora.models.diffusion.opensora_v1_3.modeling_inpaint import OpenSoraInpaint_v1_3
from transformers import T5EncoderModel, T5Tokenizer, AutoTokenizer, MT5EncoderModel, CLIPTextModelWithProjection

def get_scheduler(args):
    kwargs = dict(
        prediction_type=args.prediction_type, 
        rescale_betas_zero_snr=args.rescale_betas_zero_snr, 
        timestep_spacing="trailing" if args.rescale_betas_zero_snr else 'leading', 
    )
    if args.v1_5_scheduler:
        kwargs['beta_start'] = 0.00085
        kwargs['beta_end'] = 0.0120
        kwargs['beta_schedule'] = "scaled_linear"
    if args.sample_method == 'DDIM':  
        scheduler_cls = DDIMScheduler
        kwargs['clip_sample'] = False
    elif args.sample_method == 'EulerDiscrete':
        scheduler_cls = EulerDiscreteScheduler
    elif args.sample_method == 'DDPM':  
        scheduler_cls = DDPMScheduler
        kwargs['clip_sample'] = False
    elif args.sample_method == 'DPMSolverMultistep':
        scheduler_cls = DPMSolverMultistepScheduler
    elif args.sample_method == 'DPMSolverSinglestep':
        scheduler_cls = DPMSolverSinglestepScheduler
    elif args.sample_method == 'PNDM':
        scheduler_cls = PNDMScheduler
        kwargs.pop('rescale_betas_zero_snr', None)
    elif args.sample_method == 'HeunDiscrete':  ########
        scheduler_cls = HeunDiscreteScheduler
    elif args.sample_method == 'EulerAncestralDiscrete':
        scheduler_cls = EulerAncestralDiscreteScheduler
    elif args.sample_method == 'DEISMultistep':
        scheduler_cls = DEISMultistepScheduler
        kwargs.pop('rescale_betas_zero_snr', None)
    elif args.sample_method == 'KDPM2AncestralDiscrete':  #########
        scheduler_cls = KDPM2AncestralDiscreteScheduler
    elif args.sample_method == 'CogVideoX':
        scheduler_cls = CogVideoXDDIMScheduler
    elif args.sample_method == 'FlowMatchEulerDiscrete':
        scheduler_cls = FlowMatchEulerDiscreteScheduler
        kwargs = {}
    else:
        raise NameError(f'Unsupport sample_method {args.sample_method}')
    scheduler = scheduler_cls(**kwargs)
    return scheduler

def prepare_pipeline(args, dtype, device):
    
    weight_dtype = dtype
    print("Start to load VAE...")
    vae = ae_wrapper[args.ae](args.ae_path)
    vae.vae = vae.vae.to(device=device, dtype=weight_dtype).eval()
    vae.vae_scale_factor = ae_stride_config[args.ae]
    if args.enable_tiling:
        vae.vae.enable_tiling()

    print("Start to load TextEncoder...")
    if 'mt5' in args.text_encoder_name_1:
        text_encoder_1 = MT5EncoderModel.from_pretrained(
            args.text_encoder_name_1, cache_dir=args.cache_dir, 
            torch_dtype=weight_dtype
            ).eval()
    else:
        text_encoder_1 = T5EncoderModel.from_pretrained(
            args.text_encoder_name_1, cache_dir=args.cache_dir, 
            torch_dtype=weight_dtype
            ).eval()
    tokenizer_1 = AutoTokenizer.from_pretrained(
        args.text_encoder_name_1, cache_dir=args.cache_dir
        )

    if args.text_encoder_name_2 is not None:
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            args.text_encoder_name_2, cache_dir=args.cache_dir, 
            torch_dtype=weight_dtype
            ).eval()
        tokenizer_2 = AutoTokenizer.from_pretrained(
            args.text_encoder_name_2, cache_dir=args.cache_dir
            )
    else:
        text_encoder_2, tokenizer_2 = None, None

    print("Start to load Transformer Model...")
    if args.version == 'v1_3':
        if args.model_type in ['inpaint', 'i2v', 'iterative']:
            transformer_model = OpenSoraInpaint_v1_3.from_pretrained(
                args.model_path, cache_dir=args.cache_dir,
                device_map=None, torch_dtype=weight_dtype
                ).eval()
        else:
            transformer_model = OpenSoraT2V_v1_3.from_pretrained(
                args.model_path, cache_dir=args.cache_dir,
                device_map=None, torch_dtype=weight_dtype
                ).eval()
    elif args.version == 'v1_5':
        if args.model_type == 'inpaint' or args.model_type == 'i2v':
            raise NotImplementedError('Inpainting model is not available in v1_5')
        else:
            from opensora.models.diffusion.opensora_v1_5.modeling_opensora import OpenSoraT2V_v1_5
            transformer_model = OpenSoraT2V_v1_5.from_pretrained(
                args.model_path, cache_dir=args.cache_dir, 
                # device_map=None, 
                torch_dtype=weight_dtype
                ).eval()
    
    scheduler = get_scheduler(args)
    
    if args.model_type in ['i2v', 'inpaint']:
        pipeline_class = OpenSoraInpaintPipeline 
    elif args.model_type == 'iterative':
        pipeline_class = OpenSoraIterativePipeline
    else:
        pipeline_class = OpenSoraPipeline

    print("Start to prepare pipeline:", pipeline_class)
    pipeline = pipeline_class(
        vae=vae,
        text_encoder=text_encoder_1,
        tokenizer=tokenizer_1,
        scheduler=scheduler,
        transformer=transformer_model, 
        text_encoder_2=text_encoder_2,
        tokenizer_2=tokenizer_2,
    ).to(device)

    if args.save_memory:
        # print('enable_model_cpu_offload AND enable_sequential_cpu_offload AND enable_tiling')
        # pipeline.enable_model_cpu_offload()
        # pipeline.enable_sequential_cpu_offload()
        # torch.cuda.empty_cache()
        vae.vae.enable_tiling()
        vae.vae.t_chunk_enc = 8
        vae.vae.t_chunk_dec = vae.vae.t_chunk_enc // 2
        
    if args.compile:
        pipeline.transformer = torch.compile(pipeline.transformer)
    
    print("Pipeline is ready!")
    return pipeline

def init_gpu_env(args):
    local_rank = int(os.getenv('RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    args.local_rank = local_rank
    args.world_size = world_size
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend='nccl', init_method='env://', 
        world_size=world_size, rank=local_rank
        )
    if args.sp:
        initialize_sequence_parallel_state(world_size)
    return args

def init_npu_env(args):
    local_rank = int(os.getenv('RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    args.local_rank = local_rank
    args.world_size = world_size
    torch_npu.npu.set_device(local_rank)
    dist.init_process_group(
        backend='hccl', init_method='env://', 
        world_size=world_size, rank=local_rank
        )
    if args.sp:
        initialize_sequence_parallel_state(world_size)
    return args


def save_video_grid(video, nrow=None):
    b, t, h, w, c = video.shape

    if nrow is None:
        nrow = math.ceil(math.sqrt(b))
    ncol = math.ceil(b / nrow)
    padding = 1
    video_grid = torch.zeros(
        (
            t, 
            (padding + h) * nrow + padding, 
            (padding + w) * ncol + padding, 
            c
        ), 
        dtype=torch.uint8
        )

    for i in range(b):
        r = i // ncol
        c = i % ncol
        start_r = (padding + h) * r
        start_c = (padding + w) * c
        video_grid[:, start_r:start_r + h, start_c:start_c + w] = video[i]

    return video_grid


def run_model_and_save_samples(args, pipeline, caption_refiner_model=None, enhance_video_model=None):
    if args.seed is not None:
        set_seed(args.seed, rank=args.local_rank, device_specific=True)
    if args.local_rank >= 0:
        torch.manual_seed(args.seed + args.local_rank)
    if not os.path.exists(args.save_img_path):
        os.makedirs(args.save_img_path, exist_ok=True)

    video_grids = []
    if not isinstance(args.text_prompt, list):
        args.text_prompt = [args.text_prompt]
    if len(args.text_prompt) == 1 and args.text_prompt[0].endswith('txt'):
        text_prompt = open(args.text_prompt[0], 'r').readlines()
        args.text_prompt = [i.strip() for i in text_prompt]
    
    if args.model_type in ['inpaint', 'i2v', 'iterative']:
        if not isinstance(args.conditional_pixel_values_path, list):
            args.conditional_pixel_values_path = [args.conditional_pixel_values_path]
        if len(args.conditional_pixel_values_path) == 1 and args.conditional_pixel_values_path[0].endswith('txt'):
            temp = open(args.conditional_pixel_values_path[0], 'r').readlines()
            conditional_pixel_values_path = [i.strip().split(',') for i in temp]
        
        mask_type = args.mask_type if args.mask_type is not None else None

    positive_prompt = """
    high quality, high aesthetic, {}
    """

    negative_prompt = """
    nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, 
    low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry.
    """
    
    def generate(prompt, conditional_pixel_values_path=None, mask_type=None):
        
        if args.caption_refiner is not None:
            if args.model_type != 'inpaint' and args.model_type != 'i2v':
                refine_prompt = caption_refiner_model.get_refiner_output(prompt)
                print(f'\nOrigin prompt: {prompt}\n->\nRefine prompt: {refine_prompt}')
                prompt = refine_prompt
            else:
                # Due to the current use of LLM as the caption refiner, additional content that is not present in the control image will be added. Therefore, caption refiner is not used in this mode.
                print('Caption refiner is not available for inpainting model, use the original prompt...')
                time.sleep(3)
        input_prompt = positive_prompt.format(prompt)
        
        if args.model_type == 'inpaint' or args.model_type == 'i2v':
            print(f'\nConditional pixel values path: {conditional_pixel_values_path}')
            videos = pipeline(
                conditional_pixel_values_path=conditional_pixel_values_path,
                mask_type=mask_type,
                crop_for_hw=args.crop_for_hw,
                max_hxw=args.max_hxw,
                noise_strength=args.noise_strength,
                prompt=input_prompt, 
                negative_prompt=negative_prompt, 
                num_frames=args.num_frames,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_sampling_steps,
                guidance_scale=args.guidance_scale,
                num_samples_per_prompt=args.num_samples_per_prompt,
                max_sequence_length=args.max_sequence_length,
            ).videos
        else:
            videos = pipeline(
                input_prompt, 
                negative_prompt=negative_prompt, 
                num_frames=args.num_frames,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_sampling_steps,
                guidance_scale=args.guidance_scale,
                num_samples_per_prompt=args.num_samples_per_prompt,
                max_sequence_length=args.max_sequence_length,
            ).videos
        if enhance_video_model is not None:
            # b t h w c
            videos = enhance_video_model.enhance_a_video(videos, input_prompt, 2.0, args.fps, 250)
        if (not args.sp) or (args.sp and args.local_rank <= 0):
            if args.num_frames == 1:
                videos = rearrange(videos, 'b t h w c -> (b t) c h w')
                if args.num_samples_per_prompt != 1:
                    for i, image in enumerate(videos):
                        save_image(
                            image / 255.0, 
                            os.path.join(
                                args.save_img_path, 
                                f'{args.sample_method}_{index}_gs{args.guidance_scale}_s{args.num_sampling_steps}_i{i}.jpg'
                                ),
                            nrow=math.ceil(math.sqrt(videos.shape[0])), 
                            normalize=True, 
                            value_range=(0, 1)
                            )  # b c h w
                save_image(
                    videos / 255.0, 
                    os.path.join(
                        args.save_img_path, 
                        f'{args.sample_method}_{index}_gs{args.guidance_scale}_s{args.num_sampling_steps}.jpg'
                        ),
                    nrow=math.ceil(math.sqrt(videos.shape[0])), 
                    normalize=True, 
                    value_range=(0, 1)
                    )  # b c h w
            else:
                if args.num_samples_per_prompt == 1:
                    imageio.mimwrite(
                        os.path.join(
                            args.save_img_path,
                            f'{args.sample_method}_{index}_gs{args.guidance_scale}_s{args.num_sampling_steps}.mp4'
                        ), 
                        videos[0],
                        fps=args.fps, 
                        quality=6
                        )  # highest quality is 10, lowest is 0
                else:
                    for i in range(args.num_samples_per_prompt):
                        imageio.mimwrite(
                            os.path.join(
                                args.save_img_path,
                                f'{args.sample_method}_{index}_gs{args.guidance_scale}_s{args.num_sampling_steps}_i{i}.mp4'
                            ), videos[i],
                            fps=args.fps, 
                            quality=6
                            )  # highest quality is 10, lowest is 0
                        
                    videos = save_video_grid(videos)
                    imageio.mimwrite(
                        os.path.join(
                            args.save_img_path,
                            f'{args.sample_method}_{index}_gs{args.guidance_scale}_s{args.num_sampling_steps}.mp4'
                        ), 
                        videos,
                        fps=args.fps, 
                        quality=6
                        )  # highest quality is 10, lowest is 0)
                    videos = videos.unsqueeze(0) # 1 t h w c
            video_grids.append(videos)

    def generate_long_videos(prompt, conditional_pixel_values_path=None, mask_type=None, prompt_index=0, num_chunks=5):
        
        if args.caption_refiner is not None:
            if args.model_type != 'inpaint' and args.model_type != 'i2v':
                refine_prompt = caption_refiner_model.get_refiner_output(prompt)
                print(f'\nOrigin prompt: {prompt}\n->\nRefine prompt: {refine_prompt}')
                prompt = refine_prompt
            else:
                # Due to the current use of LLM as the caption refiner, additional content that is not present in the control image will be added. Therefore, caption refiner is not used in this mode.
                print('Caption refiner is not available for inpainting model, use the original prompt...')
                time.sleep(3)
        input_prompt = positive_prompt.format(prompt)

        long_video_dir = os.path.join(args.save_img_path, f'long_videos')
        os.makedirs(long_video_dir, exist_ok=True)

        for n_chunk in range(5, num_chunks + 1, 5):
            long_video_n_dir = os.path.join(args.save_img_path, f'long_videos_{n_chunk}_chunks')
            os.makedirs(long_video_n_dir, exist_ok=True)
        

        prompt_dir = os.path.join(args.save_img_path, f'prompt_{prompt_index}_{prompt[:50]}')
        os.makedirs(prompt_dir, exist_ok=True)

        video_chunks = []
        noise_preds_chunkwise = []
        # gamma_schedule = np.linspace(0.5, 0.05, args.num_sampling_steps)
        # gamma_schedule = torch.linspace(0.5, 0.05, args.num_sampling_steps).to(args.device)
        gamma_schedule = 0.25

        print(f'\nConditional pixel values path: {conditional_pixel_values_path}')

        for i_chunk in range(num_chunks):

            chunk_dir = os.path.join(prompt_dir, f'chunk_{i_chunk}')
            os.makedirs(chunk_dir, exist_ok=True)

            if i_chunk == 0:
                conditional_pixel_values_input = None
                cond_image = Image.open(conditional_pixel_values_path[0])
                prev_chunk_noises_every_step = None
            else:
                conditional_pixel_values_input = [video_chunks[-1][-1]]
                conditional_pixel_values_path = None
                cond_image = Image.fromarray(video_chunks[-1][-1].numpy())
                prev_chunk_noises_every_step = noise_preds_chunkwise[-1] # 0 use first chunk, -1 use last chunk            
                
            init_noises = [torch.randn((1, 8, 24, 52, 68)) for _ in range(5)]
            k_step = 8

            candidate_videos = []

            best_noise_index = 0

            if args.kstep:
                # use like 5 init noise to generate k_step samples, find best noise
                for j_candidate, noise in enumerate(init_noises):

                    cand_samples = pipeline(
                        conditional_pixel_values_path=conditional_pixel_values_path,
                        conditional_pixel_values_input=conditional_pixel_values_input,
                        mask_type=mask_type,
                        crop_for_hw=args.crop_for_hw,
                        max_hxw=args.max_hxw,
                        # noise_strength=args.noise_strength,
                        prompt=input_prompt, 
                        negative_prompt=negative_prompt, 
                        num_frames=args.num_frames,
                        height=args.height,
                        width=args.width,
                        num_inference_steps=k_step,
                        guidance_scale=args.guidance_scale,
                        num_samples_per_prompt=args.num_samples_per_prompt,
                        max_sequence_length=args.max_sequence_length,
                        latents=noise,
                        seed_for_step=42, # to make sure same noise input -> same output
                    ) # t h w c, torch uint8

                    cand_videos = cand_samples['videos'][0]
                    candidate_videos.append(cand_videos)
                    torchvision.io.write_video(os.path.join(chunk_dir, f'candidate_{j_candidate}.mp4'), cand_videos, args.fps)


                # TODO: ...all kinds of metrics, find best candidate_videos and corresponding noise
                cand_v_embeds = []
                for cand_video in candidate_videos:
                    processed_video = processor(images=cand_video, return_tensors="pt")['pixel_values'].to(args.device)
                    video_embed = clip_model(processed_video).image_embeds
                    cand_v_embeds.append(video_embed)
                
                processed_cond_image = processor(images=cond_image, return_tensors="pt")['pixel_values'].to(args.device)
                cond_image_embed = clip_model(processed_cond_image).image_embeds # [1, 512]
                cond_image_embed = cond_image_embed.repeat(93, 1) # [93, 512]

                similarities = [torch.nn.functional.cosine_similarity(cond_image_embed, cand_v_embed).min().item() for cand_v_embed in cand_v_embeds]
                similarities = torch.tensor(similarities)

                best_noise_index = similarities.argmax().item()
                print("Choose best noise:", best_noise_index)

            # actually generate this chunk, full step
            best_noise = init_noises[best_noise_index]

            samples = pipeline(
                conditional_pixel_values_path=conditional_pixel_values_path,
                conditional_pixel_values_input=conditional_pixel_values_input,
                mask_type=mask_type,
                crop_for_hw=args.crop_for_hw,
                max_hxw=args.max_hxw,
                # noise_strength=args.noise_strength,
                prompt=input_prompt, 
                negative_prompt=negative_prompt, 
                num_frames=args.num_frames,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_sampling_steps,
                guidance_scale=args.guidance_scale,
                num_samples_per_prompt=args.num_samples_per_prompt,
                max_sequence_length=args.max_sequence_length,
                latents=best_noise,
                seed_for_step=42, # to make sure same noise input -> same output
                prev_chunk_model_output_every_step=prev_chunk_noises_every_step,
                gamma_schedule=gamma_schedule,
            )

            videos = samples['videos'][0] # t h w c, torch uint8
            noise_every_step_dict = samples['noise_every_step_dict']
            noise_preds_chunkwise.append(noise_every_step_dict)

            video_chunks.append(videos)
            torchvision.io.write_video(os.path.join(chunk_dir, f'chunk_{i_chunk}.mp4'), videos, args.fps)

            if (i_chunk + 1) % 5 == 0:
                long_video_temp = torch.cat(video_chunks, dim=0)
                long_video_n_dir = os.path.join(args.save_img_path, f'long_videos_{i_chunk + 1}_chunks')
                if not os.path.exists(long_video_n_dir):
                    os.makedirs(long_video_n_dir, exist_ok=True)
                torchvision.io.write_video(os.path.join(long_video_n_dir, f'long_video_{prompt_index}_{prompt[:50]}_chunk_{i_chunk}.mp4'), long_video_temp, fps=8)
        
        long_video = torch.cat(video_chunks, dim=0)
        torchvision.io.write_video(os.path.join(long_video_dir, f'long_video_{prompt_index}_{prompt[:50]}.mp4'), long_video, fps=8)




    if args.model_type == "iterative":
        print("Start to generate iterative videos...")
        processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(args.device)
        clip_model.eval()

        total_batch = args.total_batch
        batch_idx = args.batch_idx
        assert total_batch > 0 and batch_idx >= 0 and batch_idx < total_batch

        total_num = len(args.text_prompt)


        batch_size = total_num // total_batch
        start_idx = batch_size * batch_idx
        end_idx = total_num if batch_idx == total_batch - 1 else batch_size * (batch_idx + 1)


        # ranges = np.linspace(513, 600, total_batch+1).astype(int)
        # start_idx = ranges[batch_idx]
        # end_idx = ranges[batch_idx+1]

        print("In this batch, start from {} to {}".format(start_idx, end_idx))

        bad_prompt_indices = [249, 297, 406, 457, 473, 522] # with pure black image, maybe due to some NSFW reason
        test_indices = [246, 475, 711, 258, 222, 156, 798, 216, 107, 29, 217, 290, 756, 121, 691, 749, 549, 432, 440, 438, 330, 628, 294, 738, 136, 523, 574, 745, 640, 288, 417, 338, 458, 485, 743, 218, 693, 191, 669, 399, 503, 1, 253, 65, 284, 605, 579, 123, 512, 507, 498, 470, 754, 198, 741, 469, 122, 205, 213, 68, 562, 476, 717, 680, 544, 460, 77, 771, 351, 13, 698, 144, 734, 780, 416, 468, 558, 71, 84, 40, 36, 455, 271, 251, 637, 220, 325, 791, 619, 8, 332, 532, 153, 392, 124, 46, 237, 514, 341, 718]

        print("length text_prompt:", len(args.text_prompt))
        print("length conditional_pixel_values_path:", len(conditional_pixel_values_path))
        # for index, (prompt, cond_path) in enumerate(zip(args.text_prompt[start_idx:end_idx], conditional_pixel_values_path[start_idx:end_idx])):
        for index in test_indices[:5]:
            if not args.sp and args.local_rank != -1 and index % args.world_size != args.local_rank:
                continue
            # prompt_index = start_idx + index
            
            prompt_index = index
            prompt = args.text_prompt[prompt_index]
            cond_path = conditional_pixel_values_path[prompt_index]

            if prompt_index in bad_prompt_indices:
                print("Skip bad prompt index:", prompt_index)
                continue
            if prompt_index not in test_indices:
                continue

            print(prompt_index, prompt, cond_path)
            generate_long_videos(prompt, cond_path, mask_type, num_chunks=20, prompt_index=prompt_index)

    elif args.model_type in ['inpaint', 'i2v']:
        for index, (prompt, cond_path) in enumerate(zip(args.text_prompt, conditional_pixel_values_path)):
            if not args.sp and args.local_rank != -1 and index % args.world_size != args.local_rank:
                continue
            generate(prompt, cond_path, mask_type)
    else:
        for index, prompt in enumerate(args.text_prompt):
            if index not in [1, 5, 6, 7, 8]:
                continue
            if not args.sp and args.local_rank != -1 and index % args.world_size != args.local_rank:
                continue  # skip when ddp
            generate(prompt)

    # if (args.model_type == "inpaint" or args.model_type == "i2v") and not args.crop_for_hw:
    #     print('completed, please check the saved images and videos')
    # else:
    #     if not args.sp:
    #         if args.local_rank != -1:
    #             dist.barrier()
    #             video_grids = torch.cat(video_grids, dim=0).cuda()
    #             shape = list(video_grids.shape)
    #             shape[0] *= args.world_size
    #             gathered_tensor = torch.zeros(shape, dtype=video_grids.dtype).cuda()
    #             dist.all_gather_into_tensor(gathered_tensor, video_grids.contiguous())
    #             video_grids = gathered_tensor.cpu()
    #             dist.barrier()
    #         else:
    #             video_grids = torch.cat(video_grids, dim=0)
    #     elif args.sp and args.local_rank <= 0:
    #         video_grids = torch.cat(video_grids)
        
    #     if args.local_rank <= 0:
    #         if args.num_frames == 1:
    #             save_image(
    #                 video_grids / 255.0, 
    #                 os.path.join(
    #                     args.save_img_path,
    #                     f'{args.sample_method}_gs{args.guidance_scale}_s{args.num_sampling_steps}.jpg'
    #                     ), 
    #                 nrow=math.ceil(math.sqrt(len(video_grids))), 
    #                 normalize=True, 
    #                 value_range=(0, 1)
    #                 )
    #         else:
    #             video_grids = save_video_grid(video_grids)
    #             imageio.mimwrite(
    #                 os.path.join(
    #                     args.save_img_path,
    #                     f'{args.sample_method}_gs{args.guidance_scale}_s{args.num_sampling_steps}.mp4'
    #                 ), 
    #                 video_grids, 
    #                 fps=args.fps, 
    #                 quality=6
    #                 )
    #         print('save path {}'.format(args.save_img_path))



def run_model_and_save_samples_npu(args, pipeline, caption_refiner_model=None, enhance_video_model=None):
    
    # experimental_config = torch_npu.profiler._ExperimentalConfig(
    #     profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    #     aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization
    # )
    # profile_output_path = "/home/image_data/npu_profiling_t2v"
    # os.makedirs(profile_output_path, exist_ok=True)
    # with torch_npu.profiler.profile(
    #         activities=[
    #             torch_npu.profiler.ProfilerActivity.NPU, 
    #             torch_npu.profiler.ProfilerActivity.CPU
    #             ],
    #         with_stack=True,
    #         record_shapes=True,
    #         profile_memory=True,
    #         experimental_config=experimental_config,
    #         schedule=torch_npu.profiler.schedule(
    #             wait=10000, warmup=0, active=1, repeat=1, skip_first=0
    #             ),
    #         on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(f"{profile_output_path}/")
    # ) as prof:
    run_model_and_save_samples(args, pipeline, caption_refiner_model, enhance_video_model)
        # prof.step()



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='/home/szhang3/siyang-storage/opsrplan1.3/checkpoints/any93x640x640')
    parser.add_argument("--version", type=str, default='v1_3', choices=['v1_3', 'v1_5'])
    parser.add_argument("--model_type", type=str, default='t2v', choices=['t2v', 'inpaint', 'i2v', 'iterative'])
    parser.add_argument("--num_frames", type=int, default=93)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--cache_dir", type=str, default='/home/szhang3/siyang-storage/cache_dir')
    parser.add_argument("--caption_refiner", type=str, default=None)
    parser.add_argument("--ae", type=str, default='WFVAEModel_D8_4x8x8')
    parser.add_argument("--ae_path", type=str, default='/home/szhang3/siyang-storage/opsrplan1.3/checkpoints/vae')
    parser.add_argument("--enhance_video", type=str, default=None)
    parser.add_argument("--text_encoder_name_1", type=str, default='google/mt5-xxl')
    parser.add_argument("--text_encoder_name_2", type=str, default=None)
    parser.add_argument("--save_img_path", type=str, default="/home/szhang3/siyang-storage/opsrplan1.3/sample_t2v_results2")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--sample_method", type=str, default="EulerAncestralDiscrete")
    parser.add_argument("--num_sampling_steps", type=int, default=30)
    parser.add_argument("--fps", type=int, default=18)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--text_prompt", nargs='+', default="/home/szhang3/siyang-storage/opsrplan1.3/examples/all_prompts.txt")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_samples_per_prompt", type=int, default=1)
    parser.add_argument('--enable_tiling', action='store_true')
    parser.add_argument('--refine_caption', action='store_true')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--save_memory', action='store_true') 
    parser.add_argument("--prediction_type", type=str, default='v_prediction', help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.")
    parser.add_argument('--rescale_betas_zero_snr', action='store_true')
    parser.add_argument('--local_rank', type=int, default=-1)    
    parser.add_argument('--world_size', type=int, default=1)    
    parser.add_argument('--sp', action='store_true')

    parser.add_argument('--v1_5_scheduler', action='store_true')
    parser.add_argument('--conditional_pixel_values_path', type=str, default="/home/szhang3/siyang-storage/opsrplan1.3/examples/cond_pix_path_vbench.txt")
    parser.add_argument('--mask_type', type=str, default=None)
    parser.add_argument('--crop_for_hw', action='store_true')
    parser.add_argument('--max_hxw', type=int, default=236544) # 480*480
    parser.add_argument('--noise_strength', type=float, default=0.0)
    parser.add_argument('--total_batch', type=int, default=1)
    parser.add_argument('--batch_idx', type=int, default=0)
    parser.add_argument('--kstep', action='store_true')

    args = parser.parse_args()
    assert not (args.sp and args.num_frames == 1)
    args.enable_tiling = True
    args.save_memory = True
    args.rescale_betas_zero_snr = True
    # args.compile = True
    return args


if __name__ == "__main__":
    args = get_args()
    dtype = torch.float16
    device = torch.cuda.current_device()
    enhance_video_model = None
    
    pipeline = prepare_pipeline(args, dtype, device)
    # pipeline = None
    caption_refiner_model = None

    run_model_and_save_samples(args, pipeline, caption_refiner_model, enhance_video_model)
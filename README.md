# K-step Search
Official Implementation of the paper [Towards Chunk-Wise Generation for Long Videos](https://arxiv.org/abs/2411.18668)

<a href='https://arxiv.org/abs/2411.18668'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>

Thanks for open-sourced projects [Open-Sora-Plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan), [CogVideoX](https://github.com/THUDM/CogVideo) and [StableVideoDiffusion](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt).


## Requirements

1. Clone this repository and navigate to its directory.
```
git clone https://github.com/szhan227/kstep_search.git
cd kstep_search
```

2. Setup environment
* Python >= 3.10
* Pytorch >= 2.5.1
* CUDA Version >= 12.4

```
conda create -n opensora python=3.10 -y
conda activate opensora
pip install -r requirements.txt
```

## Inference
Run the following code to generate a long video. See parse_arg() function in kstep_search.py and config files in ./configs.
```
python kstep_search.py --pipe SVD --num_chunks 5
```


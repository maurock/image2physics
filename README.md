# Image2Real

This repository provides a framework to reconstruct individual 3D meshes of a scene using user prompts and multi-view RGB images. It builds upon the ideas presented in the NeRF2Physics paper (MIT License). The current codebase has been _substantially expanded and modified_, retaining only two scripts from the original repository (`utils.py` and `arguments.py`).

# Table of Contents
- [Overview](#overview)
- [High-level Procedure](#high-level-procedure)
- [Installation](#installation)
- [License](#license)
- [Citation](#citation)

# Overview
Image2Real enables the reconstruction of 3D scenes from multi-view RGB images and user-defined prompts, providing outputs that are compatible with physics simulation environments. This project is designed for researchers and engineers interested in using NeRF-based methodologies for downstream tasks such as physical simulation. This is a work-in-progress.

# High-level Procedure
### Folder structure
If you have multi-view RGB images and corresponding camera parameters,  place them under `data/scenes/<SCENE_NAME>`. The required folder structure is:
```
root
 ├── data/
 │   ├── scenes/
 │   │   ├── <SCENE_NAME>/ 
 |   |   |   ├── images/  
 |   |   |   |   ├── cam0.png
 |   |   |   |   ├── cam1.png
 ...
 |   |   |   ├── transforms.json
```
  Please [check here](#camera-parameters) for more information on `transforms.json`.
### Data generation
 If you don't have RGB images and camera parameters, you can generate synthetic data using MuJoCo:
  - `simulator.mujoco_to_nerfstudio.py`: This scripts creates multi-view RGB images and corresponding camera parameters from a pre-defined MuJoCo scene. The generated data is stored in `data/scenes/<SCENE_NAME>`. The result consists in RGB images (`data/scenes/<SCENE_NAME>/images/`) and camera parameters (`transforms.json`), which is automatically parsed into the format required by `nerfstudio`. The required MuJoCo XML scene definitions are located in `data/mj_models`, while assets are stored in `data/assets`.

Example of usage with the Demo scene available in this repository:
 ```[python]
python simulator/mujoco_to_nerfstudio.py --radius 1 --scene_name my_scene --num_cameras 50
 ```
### Camera parameters
The format of the camera parameters required by [`nerfstudio`](https://docs.nerf.studio/) can be found in the [NeRFStudio documentation](https://docs.nerf.studio/quickstart/data_conventions.html). Please note that `nerfstudio` supports multiple commonly-used formats (e.g. `Blender`, `ScanNet`, `ARKitScenes`, `D-NeRF`). For details on available data parsers, check the [DataParser documentation](https://docs.nerf.studio/developer_guides/pipelines/dataparsers.html#dataparsers). 

### Scene reconstruction
- `reconstruction.py`: This script reconstructs a scene from the RGB images and corresponding `transforms.json` file located in `data/scenes/<SCENE_NAME>`. It uses the  [`nerfstudio`](https://docs.nerf.studio/) framework, which is open-source and licensed under [Apache License 2.0](https://github.com/nerfstudio-project/nerfstudio/blob/main/LICENSE). 
The reconstruction process uses the object names specified in the `args.text_prompt` (format: "<OBJECT>.<OBJECT>.") to segment and reconstruct thee desired object. The RGB images are segmented using [GroundingDINOv2](https://github.com/IDEA-Research/GroundingDINO) + [SAMv2](https://github.com/facebookresearch/sam2). These resulting segmentation masks are applied to depth maps predicted by NeRF, allowing for the extraction of per-object point clouds. The extracted point clouds are then reconstructed into meshes using [Poisson Surface Reconstruction](https://www.open3d.org/docs/latest/tutorial/Advanced/surface_reconstruction.html#Poisson-surface-reconstruction). Since the point clouds are already in their correct world-frame poses, no further pose optimisation is required.

Example of usage:
```
python reconstruct.py --text_prompt banana.mug. --data_dir 'data/' --end_idx 1
``` 

# Installation
In this environment we use:
- python 3.10
- CUDA 11.8, cudnn 8.8

```
conda create -n phys python=3.10
conda activate phys
```

Libraries:
```
pip install -r requirements.txt
pip install mujoco obj2mjcf mediapy
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c conda-forge ffmpeg=4.2.2
pip install -e .
```

To install Grounded-SAM2:
```
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
cd Grounded-SAM-2/
pip install -e .
```
To install [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn):
```
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```
If you get an installation error, try [this](https://github.com/NVlabs/tiny-cuda-nn/issues/183#issuecomment-1342828785). In my case, I changed `/cuda/` with `/cuda-11.8/`.


**Known issues and solutions**: 
- `pytorch3d` downgrades `jax` and `jaxlib` to versions that are not compatible with `mujoco-mjx`. The goal is to install `pytorch3d` first, and then install with the desired versions of `pytorch`, `jax`, and `jaxlib` later. This is only required for advanced tasks, like differentiable physics using the extracted meshes on MuJoCo MJX.
- The `ffmpeg` version installed with `pytorch`/`pytorch3d` is not compatible with `mediapy`. Downgrade `ffmpeg` to 4.2.2
- Error: `Loaded runtime CuDNN library: X.X.X but source was compiled with: Y.Y.Y`. This is due to the interactions between `pytorch` and `pytorch3d` with `jax` and `jaxlib`. There are various options that I am aware of to solve this: A) `pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html` B) Re-install `jaxlib` and `jax` from conda-forge: `conda install conda-forge::jaxlib` and `conda install jax -c conda-forge`. C) Check what CuDNN library you are using and refer to https://storage.googleapis.com/jax-releases/jax_cuda_releases.html to find a matching `jaxlib` distribution.

# License
The code and model provided herein are available for usage as specified in the [LICENSE](https://github.com/maurock/image2physics/blob/master/LICENSE) file (MIT License). The scripts that are based on NeRF2Physics are available for usage as specified in the [LICENSE](https://github.com/ajzhai/NeRF2Physics/blob/master/LICENSE) file (MIT License).

# Citation
Two scripts (`utils.py`, `arguments.py`) are currently based on the repository "NeRF2Physics" that Zhai et al. open-sourced on Github. Please cite their paper if you find this repo useful.

```bibtex
@inproceedings{zhai2024physical,
  title={Physical Property Understanding from Language-Embedded Feature Fields},
  author={Zhai, Albert J and Shen, Yuan and Chen, Emily Y and Wang, Gloria X and Wang, Xinlei and Wang, Sheng and Guan, Kaiyu and Wang, Shenlong},
  booktitle={CVPR},
  year={2024}
}
```
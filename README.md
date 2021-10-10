# VQGAN-CLIP-GENERATOR Overview

A package for running VQGAN+CLIP locally. This package was a complete refactor of the code provided by [NerdyRodent](https://github.com/nerdyrodent/), which started out as a Katherine Crowson VQGAN+CLIP derived Google colab notebook.

In addition to refactoring NerdyRodent's code into a pythonic package to improve usability, this project adds unit tests, and adds improvements to the ability to restyle an existing video.

Original notebook: [![Open In Colab][colab-badge]][colab-notebook]

[colab-notebook]: <https://colab.research.google.com/drive/1ZAus_gn2RhTZWzOWUpPERNC0Q8OhZRTZ>
[colab-badge]: <https://colab.research.google.com/assets/colab-badge.svg>

[NerdyRodent VQGAN+CLIP repository](https://github.com/nerdyrodent/)

Some example images:

<img src="./samples/A child throwing the ducks into a wood chipper painting by Rembrandt initial.png" width="256px"></img>
<img src="./samples/Pastoral landscape painting in the impressionist style initial.png" width="256px"></img>
<img src="./samples/The_sadness_of_Colonel_Sanders_by_Thomas_Kinkade.png" width="256px"></img>

Environment:

* Tested on Windows 10 build 19043
* GPU: Nvidia RTX 3080
* CPU: AMD 5900X
* Typical VRAM requirements:
  * 24 GB for a 900x900 image
  * 10 GB for a 512x512 image
  * 8 GB for a 380x380 image

## Setup
### Virtual environment
This example uses [Anaconda](https://www.anaconda.com/products/individual#Downloads) to manage virtual Python environments. Create a new virtual Python environment for VQGAN-CLIP-GENERATOR. Then, install the VQGAN-CLIP-GENERATOR package using pip.

```sh
conda create --name vqgan python=3.9 pip ffmpeg numpy pytest tqdm git pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
conda activate vqgan
pip install git+https://github.com/openai/CLIP.git taming-transformers ftfy regex tqdm pytorch-lightning kornia imageio omegaconf taming-transformers torch_optimizer
pip install vqgan-clip-generator
```

### Download model

The VQGAN algorithm requires use of a compatible model file. These files are not provided with the pip intallation, and must be downloaded separately. You can either download them manually, or use the provided download method. Note that when using this package you must specify the location where you've saved these model files.

```sh
mkdir models

curl -L -o models/vqgan_imagenet_f16_16384.yaml -C - 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1' #ImageNet 16384
curl -L -o models/vqgan_imagenet_f16_16384.ckpt -C - 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1' #ImageNet 16384
```

Or, using the provided download method

```python
vqgan_clip.download(".\models\")
``` 
### Quick example to confirm that it works
```python
import vqgan_clip.generate
from vqgan_clip.engine import VQGAN_CLIP_Config

config = VQGAN_CLIP_Config()
config.vqgan_config = f'models/vqgan_imagenet_f16_16384.yaml'
config.vqgan_checkpoint = f'models/vqgan_imagenet_f16_16384.ckpt'
config.text_prompts = 'A portrait of a man by Rembrandt'
vqgan_clip.generate.single_image(config)
```

### If using an AMD graphics card

The instructions above assume an nvidia GPU with support for CUDA 11.1. Instructions for an AMD GPU below are courtesy of NerdyRodent. Note: I have not tested this advice.

ROCm can be used for AMD graphics cards instead of CUDA. You can check if your card is supported here:
<https://github.com/RadeonOpenCompute/ROCm#supported-gpus>

Install ROCm accordng to the instructions and don't forget to add the user to the video group:
<https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html>

The usage and set up instructions above are the same, except for the line where you install Pytorch.
Instead of `pip install torch==1.9.0+cu111 ...`, use the one or two lines which are displayed here (select Pip -> Python-> ROCm):
<https://pytorch.org/get-started/locally/>

### If using the CPU

If no graphics card can be found, the CPU is automatically used and a warning displayed.

Regardless of an available graphics card, the CPU can also be used by adding this command line argument: `-cd cpu`

This works with the CUDA version of Pytorch, even without CUDA drivers installed, but doesn't seem to work with ROCm as of now.

### Uninstalling

Remove the Python enviroment:

```sh
conda deactivate
conda remove --name vqgan --all
```
## Generating images and video

### Prompts
Prompts are objects that can be analyzed by CLIP to identify their contents. The resulting images will be those that are similar to the prompts, as evaluated by CLIP. Prompts can be any combination of text phrases, example images, or random number generator seeds. Each of these types of prompts is in a separate string, discussed below.

Multiple prompts can be combined, both in parallel and in series. Prompts that should be used in parallel are separated by a pipe symbol, like so:
```python
'first parallel prompt | second parallel prompt'
```
Prompts that should be processed in series should be separated by a carat (^). Serial prompts, sepated by ^, will be cycled through every change_prompt_every iterations. Prompts will loop if more cycles are requested than there are prompts. This feature is primarily intended for use when generating videos.

```python
'first serial prompt ^ second serial prompt'
```

Prompts may be given different weights by following them with ':float'. A weight of 1.0 is assumed if no value is provided.
```python
'prompt 10x more weighted:1.0 | prompt with less weight:0.1'
```

These methods may be used in any combination.
```python
'prompt 1:1.0 | prompt 2:0.1 | prompt 3:0.5 ^ prompt 4 | prompt 5 | prompt 6:2.0'
```

### Image generation parameters
The parameters used for image generation are stored in a VQGAN_CLIP_Config instance. Instantiate this class and customize the attributes as needed, then pass this configuratio object to a method of vqgan_clip.generate.
|Attribute|Default|Meaning
|---------|---------|---------|
|text_prompts|'A painting of flowers in the renaissance style:0.5\|rembrandt:0.5^fish:0.2\|love:1'|Text prompt for image generation|
|image_prompts|[]|Path to image(s) that will be turned into a prompt via CLIP. The contents of the resulting image will have simiar content to the prompt image(s) as evaluated by CLIP.|
|noise_prompts|[]|Random number seeds can be used as prompts using the same format as a text prompt. E.g. '123:0.1\|234:0.2\|345:0.\|3' Stories (^) are supported. |
|iterations|100|Number of iterations of train() to perform before stopping and outputing the image. The resulting still image will eventually converge to an image that doesn't perceptually change much in content.|
|save_every|50|An interim image will be saved to the output location every save_every iterations. If you are generating a video, a frame of video will be created every save_every iterations.|
|change_prompt_every|0|Serial prompts, sepated by ^, will be cycled through every change_prompt_every iterations. Prompts will loop if more cycles are requested than there are prompts.|
|output_image_size|[256,256]|x/y dimensions of the output image in pixels. This will be adjusted slightly based on the GAN model used. VRAM requirements increase steeply with image size. A video card with 10GB of VRAM can handle a size of [448,448]|
|output_filename|'output.png'|Location to save the output image file when a single file is being created.|
|init_image|None|A Seed image that can be used to start the training. Without an initial image, random noise will be used.|
|init_noise|None|Seed an image with noise. Options None, 'pixels' or 'gradient'|
|vqgan_config|f'models/vqgan_imagenet_f16_16384.yaml'|Path to model yaml file. This must be customized to match the location where you downloaded the model file.|
|vqgan_checkpoint|f'models/vqgan_imagenet_f16_16384.ckpt'|Path to model checkpoint file. This must be customized to match the location where you downloaded the model file.|

Other configuration attributes can be seen in vqgan_clip.engine.VQGAN_CLIP_Config. Those options are related to the function of the algorithm itself. For example, you can change the learning rate of the GAN, or change the optimization algorithm used, or change the GPU used.

## Examples
### Generating a single image from a text prompt
In the example below, an image is generated from two text prompts: "A pastoral landscape painting by Rembrandt" and "A blue fence." These prompts are given different weights during image genration, with the first weighted ten-fold more heavily than the second. This method of applying weights can be used, or not, for all three types of prompts: text, images, and noise. If a weight is not specified, a weight of 1.0 is assumed.

```python
import vqgan_clip.generate
from vqgan_clip.engine import VQGAN_CLIP_Config

config = VQGAN_CLIP_Config()
config.vqgan_config = f'models/vqgan_imagenet_f16_16384.yaml'
config.vqgan_checkpoint = f'models/vqgan_imagenet_f16_16384.ckpt'
config.text_prompts = 'A pastoral landscape painting by Rembrandt:1.0|A blue fence:0.1'
config.iterations = 200
config.output_image_size = [448,448]
vqgan_clip.generate.single_image(config)
```

## Troubleshooting

### RuntimeError: CUDA out of memory
For example:
```
RuntimeError: CUDA out of memory. Tried to allocate 150.00 MiB (GPU 0; 23.70 GiB total capacity; 21.31 GiB already allocated; 78.56 MiB free; 21.70 GiB reserved in total by PyTorch)
```
Your request doesn't fit into your GPU's VRAM. Reduce the image size and/or number of cuts.

## Citations

```bibtex
@misc{unpublished2021clip,
    title  = {CLIP: Connecting Text and Images},
    author = {Alec Radford, Ilya Sutskever, Jong Wook Kim, Gretchen Krueger, Sandhini Agarwal},
    year   = {2021}
}
```

```bibtex
@misc{esser2020taming,
      title={Taming Transformers for High-Resolution Image Synthesis}, 
      author={Patrick Esser and Robin Rombach and Björn Ommer},
      year={2020},
      eprint={2012.09841},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

Katherine Crowson - <https://github.com/crowsonkb>
NerdyRodent - <https://github.com/nerdyrodent/>

Public Domain images from Open Access Images at the Art Institute of Chicago - <https://www.artic.edu/open-access/open-access-images>
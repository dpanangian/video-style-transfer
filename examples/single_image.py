# Generate a single image based on a text prompt
from vqgan_clip import generate, esrgan
from vqgan_clip.engine import VQGAN_CLIP_Config
import os
from vqgan_clip import _functional as VF

config = VQGAN_CLIP_Config()
config.output_image_size = [587, 330]
# Set True if you installed the Real-ESRGAN package for upscaling.
upscale_image = True
text_prompts = 'A pastoral landscape painting by Rembrandt'

output_filename = f'example media{os.sep}example image.png'
metadata_comment = generate.image(eng_config=config,
                                  text_prompts=text_prompts,
                                  iterations=100,
                                  output_filename=output_filename)

# Upscale the image
if upscale_image:
    esrgan.inference_realesrgan(input=output_filename,
                                output_images_path='example media',
                                face_enhance=False,
                                netscale=4,
                                outscale=4)
    VF.copy_PNG_metadata(output_filename, os.path.splitext(output_filename)[0]+'_upscaled.png')
print(f'generation parameters: {metadata_comment}')

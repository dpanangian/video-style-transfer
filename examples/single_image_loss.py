# Generate a single image based on a text prompt
# Note that any input images or video are not provided for example scripts, you will have to provide your own.

from vqgan_clip import generate, esrgan
from vqgan_clip.engine import VQGAN_CLIP_Config
import os
from vqgan_clip import _functional as VF

config = VQGAN_CLIP_Config()
config.output_image_size = [360,640]
config.seed = 111

# Set True if you installed the Real-ESRGAN package for upscaling.
upscale_image = False
text_prompts = None
image_prompts = '/content/drive/MyDrive/vqgan-clip-generator/samples/hokusai.jpg'

for loss in ['original','mse','mse-lpips','mse-pixel-lpips','mse-pixel']:
    
    output_filename = f'example media{os.sep}example image_{loss}.jpg'
    config.init_image_method = loss
    metadata_comment = generate.image(eng_config=config,
                                    init_image=r'/content/drive/MyDrive/vqgan-clip-generator/images/init_image.jpg',
                                    init_weight=0.8,
                                    save_every=1,
                                    text_prompts=text_prompts,
                                    image_prompts=image_prompts,
                                    iterations=120,
                                    output_filename=output_filename,
                                    verbose=True)

    # Upscale the image
    if upscale_image:
        esrgan.inference_realesrgan(input=output_filename,
                                    output_images_path='example media',
                                    face_enhance=False,
                                    netscale=4,
                                    outscale=4)
        VF.copy_image_metadata(output_filename, os.path.splitext(output_filename)[0]+'_upscaled.jpg')
    print(f'generation parameters: {metadata_comment}')

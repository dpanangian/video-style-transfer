# This module is the interface for creating images and video from text prompts
# This should also serve as examples of how you can use the Engine class to create images and video using your own creativity.
# Feel free to extract the contents of these methods and use them to build your own sequences. 
# Change the image prompt weights over time
# Change the interval at which video frames are exported over time, to create the effect of speeding or slowing video
# Change the engine learning rate to increase or decrease the amount of change for each frame
# Create style transfer videos where each frame uses many image prompts, or many previous frames as image prompts.
# Create a zoom video where the shift_x and shift_x are functions of iteration to create spiraling zooms
# It's art. Go nuts!

from vqgan_clip.engine import Engine, VQGAN_CLIP_Config
from vqgan_clip.z_smoother import Z_Smoother
from tqdm.auto import tqdm
import os
import contextlib
import torch
import warnings
from PIL import ImageFile, Image, ImageChops, PngImagePlugin
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision.transforms import functional as TF
from vqgan_clip import _functional as VF
import cv2
from os import listdir
from os.path import isfile, join
import numpy as np
import shutil
import glob
from distutils.dir_util import copy_tree
import numpy as np
from PIL import Image
import sys
import math

from patch_based.train import style_transfer as st
    

def image(output_filename,
        eng_config = VQGAN_CLIP_Config(),
        text_prompts = [],
        image_prompts = [],
        noise_prompts = [],
        init_image = None,
        init_weight = 0.0,
        iterations = 100,
        save_every = None,
        verbose = False,
        leave_progress_bar = True):
    """Generate a single image using VQGAN+CLIP. The configuration of the algorithms is done via a VQGAN_CLIP_Config instance.

    Args:
        * output_filename (str) : location to save the output image. Omit the file extension. 
        * eng_config (VQGAN_CLIP_Config, optional): An instance of VQGAN_CLIP_Config with attributes customized for your use. See the documentation for VQGAN_CLIP_Config().
        * text_prompts (str, optional) : Text that will be turned into a prompt via CLIP. Default = []  
        * image_prompts (str, optional) : Path to image that will be turned into a prompt via CLIP (analyzed for content). Default = []
        * noise_prompts (str, optional) : Random number seeds can be used as prompts using the same format as a text prompt. E.g. \'123:0.1|234:0.2|345:0.3\' Stories (^) are supported. Default = []
        * init_image (str, optional) : Path to an image file that will be used as the seed to generate output (analyzed for pixels).
        * init_weight (float, optional) : Relative weight to assign to keeping the init_image content.
        * iterations (int, optional) : Number of iterations of train() to perform before stopping. Default = 100 
        * save_every (int, optional) : An interim image will be saved as the final image is being generated. It's saved to the output location every save_every iterations, and training stats will be displayed. Default = None  
        * verbose (boolean, optional) : When true, prints diagnostic data every time a video frame is saved. Defaults to False.
        * leave_progress_bar (boolean, optional) : When False, the tqdm progress bar will disappear when the work is completed. Useful for nested loops.
    """
    if text_prompts not in [[], None] and not isinstance(text_prompts, str):
        raise ValueError('text_prompts must be a string')
    if image_prompts not in [[], None] and not isinstance(image_prompts, str):
        raise ValueError('image_prompts must be a string')
    if noise_prompts not in [[], None] and not isinstance(noise_prompts, str):
        raise ValueError('noise_prompts must be a string')
    if init_image not in [[], None] and not os.path.isfile(init_image):
        raise ValueError(f'init_image does not exist.')
    if save_every not in [[], None] and not isinstance(save_every, int):
        raise ValueError(f'save_every must be an int.')
    if text_prompts in [[], None] and image_prompts in [[], None] and noise_prompts in [[], None]:
        raise ValueError('No valid prompts were provided')

    # output_filename = _filename_to_jpg(output_filename)
    output_folder_name = os.path.dirname(output_filename)
    if output_folder_name:
        os.makedirs(output_folder_name, exist_ok=True)

    if init_image:
        eng_config.init_image = init_image
        output_size_X, output_size_Y = VF.filesize_matching_aspect_ratio(init_image, eng_config.output_image_size[0], eng_config.output_image_size[1])
        eng_config.output_image_size = [output_size_X, output_size_Y]
        eng_config.init_weight = init_weight

    # suppress stdout to keep the progress bar clear
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            eng = Engine(eng_config)
            eng.initialize_VQGAN_CLIP()
    parsed_text_prompts, parsed_image_prompts, parsed_noise_prompts = VF.parse_all_prompts(text_prompts, image_prompts, noise_prompts)
    eng.encode_and_append_prompts(0, parsed_text_prompts, parsed_image_prompts, parsed_noise_prompts)
    eng.configure_optimizer()
    # metadata to save to jpge file as data chunks
    img_info =  [('text_prompts',text_prompts),
            ('image_prompts',image_prompts),
            ('noise_prompts',noise_prompts),
            ('iterations',iterations),
            ('init_image',init_image),
            ('save_every',save_every),
            ('cut_method',eng_config.cut_method),
            ('seed',eng.conf.seed)]

    # generate the image
    try:
        for iteration_num in tqdm(range(1,iterations+1),unit='iteration',desc='single image',leave=leave_progress_bar):
            #perform iterations of train()
            lossAll = eng.train(iteration_num)   
            if save_every and iteration_num % save_every == 0:
                if verbose:
                    # display some statistics about how the GAN training is going whever we save an interim image
                    losses_str = ', '.join(f'{loss.item():7.3f}' for loss in lossAll)
                    tqdm.write(f'iteration:{iteration_num:6d}\tloss sum: {sum(lossAll).item():7.3f}\tloss for each prompt:{losses_str}')
                # save an interim copy of the image so you can look at it as it changes if you like
                eng.save_current_output(output_filename,img_info) 

        # Always save the output at the end
        eng.save_current_output(output_filename,img_info) 
    except KeyboardInterrupt:
        pass


    config_info=f'iterations: {iterations}, '\
            f'image_prompts: {image_prompts}, '\
            f'noise_prompts: {noise_prompts}, '\
            f'init_weight_method: {",".join(eng_config.init_image_method)}, '\
            f'init_weight {",".join([str(x) for x in eng_config.init_weight])}, '\
            f'init_image {init_image}, '\
            f'cut_method {eng_config.cut_method}, '\
            f'seed {eng.conf.seed}'
    return config_info


def style_transfer(video_frames,
    eng_config=VQGAN_CLIP_Config(),
    text_prompts = 'Covered in spiders | Surreal:0.5',
    image_prompts = [],
    noise_prompts = [],
    iterations_per_frame = 15,
    iterations_for_first_frame = 15,
    current_source_frame_image_weight = 2.0,
    change_prompts_on_frame = None,
    generated_video_frames_path='./video_frames',
    current_source_frame_prompt_weight=0.0,
    z_smoother=False,
    z_smoother_buffer_len=3,
    z_smoother_alpha=0.7,
    verbose=False,
    leave_progress_bar = True,
    output_extension='jpg'):
    """Apply a style to existing video frames using VQGAN+CLIP.
    Set values of iteration_per_frame to determine how much the style transfer effect will be.
    Set values of source_frame_weight to determine how closely the result will match the source image. Balance iteration_per_frame and source_frame_weight to influence output.
    Set z_smoother to True to apply some latent-vector-based motion smoothing that will increase frame-to-frame consistency further at the cost of adding some motion blur.
    Set current_source_frame_prompt_weight >0 to have the generated content CLIP-match the source image.
    Args:
    * video_frames (list of str) : List of paths to the video frames that will be restyled.
    * eng_config (VQGAN_CLIP_Config, optional): An instance of VQGAN_CLIP_Config with attributes customized for your use. See the documentation for VQGAN_CLIP_Config().
    * text_prompts (str, optional) : Text that will be turned into a prompt via CLIP. Default = []  
    * image_prompts (str, optional) : Path to image that will be turned into a prompt via CLIP. Default = []
    * noise_prompts (str, optional) : Random number seeds can be used as prompts using the same format as a text prompt. E.g. \'123:0.1|234:0.2|345:0.3\' Stories (^) are supported. Default = []
    * change_prompts_on_frame (list(int)) : All prompts (separated by "^" will be cycled forward on the video frames provided here. Defaults to None.
    * iterations_per_frame (int, optional) : Number of iterations of train() to perform for each frame of video. Default = 15 
    * iterations_for_first_frame (int, optional) : Number of additional iterations of train() to perform on the first frame so that the image is not a gray/random field. Default = 30
    * generated_video_frames_path (str, optional) : Path where still images should be saved as they are generated before being combined into a video. Defaults to './video_frames'.
    * current_source_frame_image_weight (float) : Assigns a loss weight to make the output image look like the source image itself. Default = 0.0
    * current_source_frame_prompt_weight (float) : Assigns a loss weight to make the output image look like the CLIP representation of the source image. Default = 0.0
    * z_smoother (boolean, optional) : If true, smooth the latent vectors (z) used for image generation by combining multiple z vectors through an exponentially weighted moving average (EWMA). Defaults to False.
    * z_smoother_buffer_len (int, optional) : How many images' latent vectors should be combined in the smoothing algorithm. Bigger numbers will be smoother, and have more blurred motion. Must be an odd number. Defaults to 3.
    * z_smoother_alpha (float, optional) : When combining multiple latent vectors for smoothing, this sets how important the "keyframe" z is. As frames move further from the keyframe, their weight drops by (1-z_smoother_alpha) each frame. Bigger numbers apply more smoothing. Defaults to 0.6.
    * leave_progress_bar (boolean, optional) : When False, the tqdm progress bar will disappear when the work is completed. Useful for nested loops.
"""
    if text_prompts not in [[], None] and not isinstance(text_prompts, str):
        raise ValueError('text_prompts must be a string')
    if image_prompts not in [[], None] and not isinstance(image_prompts, str):
        raise ValueError('image_prompts must be a string')
    if noise_prompts not in [[], None] and not isinstance(noise_prompts, str):
        raise ValueError('noise_prompts must be a string')
    if text_prompts in [[], None] and image_prompts in [[], None] and noise_prompts in [[], None]:
        raise ValueError('No valid prompts were provided')
    if not isinstance(video_frames,list) or not os.path.isfile(f'{video_frames[0]}'):
        raise ValueError(f'video_frames must be a list of paths to files.')

    eng_config.init_weight = current_source_frame_image_weight

    # by default, run the first frame for the same number of iterations as the rest of the frames. It can be useful to use more though.
    if not iterations_for_first_frame:
        iterations_for_first_frame = iterations_per_frame

    output_size_X, output_size_Y = VF.filesize_matching_aspect_ratio(video_frames[0], eng_config.output_image_size[0], eng_config.output_image_size[1])
    eng_config.output_image_size = [output_size_X, output_size_Y]

    # Let's generate a single image to initialize the video. Otherwise it takes a few frames for the new video to stabilize on the generated imagery.
    init_image = 'init_image.jpg'
    eng_config_init_img = eng_config
    #eng_config_init_img.init_image_method = 'mse'
    image(output_filename=init_image,
        eng_config=eng_config_init_img,
        text_prompts=text_prompts,
        image_prompts = image_prompts,
        noise_prompts = noise_prompts,
        init_image = video_frames[0],
        init_weight=current_source_frame_image_weight,
        iterations = iterations_for_first_frame,
        save_every = None,
        verbose = False,
        leave_progress_bar = False)

    parsed_text_prompts, parsed_image_prompts, parsed_noise_prompts = VF.parse_all_prompts(text_prompts, image_prompts, noise_prompts)

    # lock in a seed to use for each frame
    if not eng_config.seed:
        # note, retreiving torch.seed() also sets the torch seed
        eng_config.seed = torch.seed()

    # if the location for the generated video frames doesn't exist, create it
    if not os.path.exists(generated_video_frames_path):
        os.mkdir(generated_video_frames_path)
    else:
        VF.delete_files(generated_video_frames_path)

    output_size_X, output_size_Y = VF.filesize_matching_aspect_ratio(video_frames[0], eng_config.output_image_size[0], eng_config.output_image_size[1])
    eng_config.output_image_size = [output_size_X, output_size_Y]
    # alternate_img_target is required for restyling video. alternate_img_target_decay is experimental.
    #if eng_config.init_image_method not in ['alternate_img_target_decay', 'alternate_img_target']:
    #    eng_config.init_image_method = 'alternate_img_target'

    # suppress stdout to keep the progress bar clear
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            eng = Engine(eng_config)
            eng.initialize_VQGAN_CLIP()
    eng.alternate_image = True

    if z_smoother:
        # Populate the z smoother with the initial image
        init_image_pil = Image.open(init_image).convert('RGB').resize([output_size_X,output_size_Y], resample=Image.LANCZOS)
        # init_img_z = eng.pil_image_to_latent_vector(init_image_pil)
        smoothed_z = Z_Smoother(buffer_len=z_smoother_buffer_len, alpha=z_smoother_alpha)

    # generate images
    video_frame_num = 1
    current_prompt_number = 0
    try:
        # To generate the first frame of video, either use the init_image argument, or the first frame of source video.
        pil_image_previous_generated_frame = Image.open(init_image).convert('RGB').resize([output_size_X,output_size_Y], resample=Image.LANCZOS)
        eng.convert_image_to_init_image(pil_image_previous_generated_frame)
        eng.configure_optimizer()
        video_frames_loop = tqdm(video_frames,unit='image',desc='style transfer',leave=leave_progress_bar)
        for video_frame in video_frames_loop:
            filename_to_save = os.path.basename(os.path.splitext(video_frame)[0]) + '.' + output_extension
            filepath_to_save = os.path.join(generated_video_frames_path,filename_to_save)

            # INIT IMAGE
            # Alternate aglorithm - init image is unchanged from the previous output. We are not resetting the tensor gradient.
            # alternate_image_target is the new source frame of video. Apply a loss in Engine using conf.init_image_method == 'alternate_img_target'
            # The previous output will be trained to change toward the new source frame.
            pil_image_new_frame = Image.open(video_frame).convert('RGB').resize([output_size_X,output_size_Y], resample=Image.LANCZOS)
            eng.set_alternate_image_target(pil_image_new_frame)

            # Optionally use the current source video frame, and the previous generate frames, as input prompts
            eng.clear_all_prompts()
            if change_prompts_on_frame is not None:
                if video_frame_num in change_prompts_on_frame:
                    # change prompts if the current frame number is in the list of change frames
                    current_prompt_number += 1
            eng.encode_and_append_prompts(current_prompt_number, parsed_text_prompts, parsed_image_prompts, parsed_noise_prompts)
            if current_source_frame_prompt_weight:
                eng.encode_and_append_pil_image(pil_image_new_frame, weight=current_source_frame_prompt_weight)

            # Generate a new image
            for iteration_num in tqdm(range(1,iterations_per_frame+1),unit='iteration',desc='generating frame',leave=False):
                #perform iterations of train()
                lossAll = eng.train(iteration_num)          

            if verbose:
                # display some statistics about how the GAN training is going whever we save an image
                losses_str = ', '.join(f'{loss.item():7.3f}' for loss in lossAll)
                tqdm.write(f'iteration:{iteration_num:6d}\tvideo frame: {video_frame_num:6d}\tloss sum: {sum(lossAll).item():7.3f}\tloss for each prompt:{losses_str}')

            if z_smoother:
                smoothed_z.append(eng._z.clone())
                output_tensor = eng.synth(smoothed_z._mid_ewma())
                Engine.save_tensor_as_image(output_tensor,filepath_to_save,img_info)
            else:
                eng.save_current_output(filepath_to_save,img_info)
            last_video_frame_generated = filepath_to_save
            video_frame_num += 1
    except KeyboardInterrupt:
        pass

    config_info=f'iterations_per_frame: {iterations_per_frame}, '\
            f'image_prompts: {image_prompts}, '\
            f'noise_prompts: {noise_prompts}, '\
            f'init_weight_method: {",".join(eng_config.init_image_method)}, '\
            f'init_weight {",".join([str(x) for x in eng_config.init_weight]}, '\
            f'current_source_frame_prompt_weight {",".join([str(x) for x in current_source_frame_prompt_weight])}, '\
            f'current_source_frame_image_weight {",".join([str(x) for x in current_source_frame_image_weight])}, '\
            f'cut_method {eng_config.cut_method}, '\
            f'z_smoother {z_smoother:2.2f}, '\
            f'z_smoother_buffer_len {z_smoother_buffer_len:2.2f}, '\
            f'z_smoother_alpha {z_smoother_alpha:2.2f}, '\
            f'seed {eng.conf.seed}'

    return config_info


def style_transfer_per_frame(video_frames,
    eng_config=VQGAN_CLIP_Config(),
    text_prompts = 'Covered in spiders | Surreal:0.5',
    image_prompts = [],
    noise_prompts = [],
    iterations_per_frame = 15,
    current_source_frame_image_weight = 2.0,
    generated_video_frames_path='./video_frames',
    current_source_frame_prompt_weight=0.0,
    verbose=False,
    leave_progress_bar = True,
    output_extension='jpg'):
    """Apply a style to existing video frames using VQGAN+CLIP.
    Set values of iteration_per_frame to determine how much the style transfer effect will be.
    Set values of source_frame_weight to determine how closely the result will match the source image. Balance iteration_per_frame and source_frame_weight to influence output.
    Set z_smoother to True to apply some latent-vector-based motion smoothing that will increase frame-to-frame consistency further at the cost of adding some motion blur.
    Set current_source_frame_prompt_weight >0 to have the generated content CLIP-match the source image.
    Args:
    * video_frames (list of str) : List of paths to the video frames that will be restyled.
    * eng_config (VQGAN_CLIP_Config, optional): An instance of VQGAN_CLIP_Config with attributes customized for your use. See the documentation for VQGAN_CLIP_Config().
    * text_prompts (str, optional) : Text that will be turned into a prompt via CLIP. Default = []  
    * image_prompts (str, optional) : Path to image that will be turned into a prompt via CLIP. Default = []
    * noise_prompts (str, optional) : Random number seeds can be used as prompts using the same format as a text prompt. E.g. \'123:0.1|234:0.2|345:0.3\' Stories (^) are supported. Default = []
    * change_prompts_on_frame (list(int)) : All prompts (separated by "^" will be cycled forward on the video frames provided here. Defaults to None.
    * iterations_per_frame (int, optional) : Number of iterations of train() to perform for each frame of video. Default = 15 
    * iterations_for_first_frame (int, optional) : Number of additional iterations of train() to perform on the first frame so that the image is not a gray/random field. Default = 30
    * generated_video_frames_path (str, optional) : Path where still images should be saved as they are generated before being combined into a video. Defaults to './video_frames'.
    * current_source_frame_image_weight (float) : Assigns a loss weight to make the output image look like the source image itself. Default = 0.0
    * current_source_frame_prompt_weight (float) : Assigns a loss weight to make the output image look like the CLIP representation of the source image. Default = 0.0
    * z_smoother (boolean, optional) : If true, smooth the latent vectors (z) used for image generation by combining multiple z vectors through an exponentially weighted moving average (EWMA). Defaults to False.
    * z_smoother_buffer_len (int, optional) : How many images' latent vectors should be combined in the smoothing algorithm. Bigger numbers will be smoother, and have more blurred motion. Must be an odd number. Defaults to 3.
    * z_smoother_alpha (float, optional) : When combining multiple latent vectors for smoothing, this sets how important the "keyframe" z is. As frames move further from the keyframe, their weight drops by (1-z_smoother_alpha) each frame. Bigger numbers apply more smoothing. Defaults to 0.6.
    * leave_progress_bar (boolean, optional) : When False, the tqdm progress bar will disappear when the work is completed. Useful for nested loops.
"""
    if text_prompts not in [[], None] and not isinstance(text_prompts, str):
        raise ValueError('text_prompts must be a string')
    if image_prompts not in [[], None] and not isinstance(image_prompts, str):
        raise ValueError('image_prompts must be a string')
    if noise_prompts not in [[], None] and not isinstance(noise_prompts, str):
        raise ValueError('noise_prompts must be a string')
    if text_prompts in [[], None] and image_prompts in [[], None] and noise_prompts in [[], None]:
        raise ValueError('No valid prompts were provided')
    if not isinstance(video_frames,list) or not os.path.isfile(f'{video_frames[0]}'):
        raise ValueError(f'video_frames must be a list of paths to files.')

    eng_config.init_weight = current_source_frame_image_weight

    output_size_X, output_size_Y = VF.filesize_matching_aspect_ratio(video_frames[0], eng_config.output_image_size[0], eng_config.output_image_size[1])
    eng_config.output_image_size = [output_size_X, output_size_Y]
    
    
    # lock in a seed to use for each frame
    if not eng_config.seed:
        # note, retreiving torch.seed() also sets the torch seed
        eng_config.seed = torch.seed()

    # if the location for the generated video frames doesn't exist, create it
    if not os.path.exists(generated_video_frames_path):
        os.mkdir(generated_video_frames_path)
    else:
        VF.delete_files(generated_video_frames_path)

    output_size_X, output_size_Y = VF.filesize_matching_aspect_ratio(video_frames[0], eng_config.output_image_size[0], eng_config.output_image_size[1])
    eng_config.output_image_size = [output_size_X, output_size_Y]
    
    # suppress stdout to keep the progress bar clear
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            eng = Engine(eng_config)
            eng.initialize_VQGAN_CLIP()

    # generate images
    video_frame_num = 1
    current_prompt_number = 0
    try:
        # To generate the first frame of video, either use the init_image argument, or the first frame of source video.
        video_frames_loop = tqdm(video_frames,unit='image',desc='style transfer',leave=leave_progress_bar)
        for video_frame in video_frames_loop:
            filename_to_save = os.path.basename(os.path.splitext(video_frame)[0]) + '.' + output_extension
            filepath_to_save = os.path.join(generated_video_frames_path,filename_to_save)

            # INIT IMAGE
            # Alternate aglorithm - init image is unchanged from the previous output. We are not resetting the tensor gradient.
            # alternate_image_target is the new source frame of video. Apply a loss in Engine using conf.init_image_method == 'alternate_img_target'
            # The previous output will be trained to change toward the new source frame.
           
            image(output_filename=filepath_to_save,
                eng_config=eng_config,
                text_prompts=text_prompts,
                image_prompts = image_prompts,
                noise_prompts = noise_prompts,
                init_image = video_frame,
                init_weight=current_source_frame_image_weight,
                iterations = iterations_per_frame,
                save_every = None,
                verbose = False,
                leave_progress_bar = False)           
            video_frame_num += 1
    except KeyboardInterrupt:
        pass

    config_info=f'iterations_per_frame: {iterations_per_frame}, '\
            f'image_prompts: {image_prompts}, '\
            f'noise_prompts: {noise_prompts}, '\
            f'init_weight {",".join([str(x) for x in eng_config.init_weight])}, '\
            f'current_source_frame_prompt_weight {",".join([str(x) for x in current_source_frame_prompt_weight])}, '\
            f'current_source_frame_image_weight {",".join([str(x) for x in current_source_frame_image_weight])}, '\
            f'cut_method {eng_config.cut_method}, '\
            f'seed {eng.conf.seed}'
            

    return config_info

def style_transfer_fewshot(video_frames,
    video_frames_path,
    output_path,
    fewshot_config_file,
    temp_fewshot_path=None,
    eng_config=VQGAN_CLIP_Config(),
    text_prompts = 'Covered in spiders | Surreal:0.5',
    image_prompts = [],
    noise_prompts = [],
    iterations_per_frame = 15,
    current_source_frame_image_weight = 2.0,
    current_source_frame_prompt_weight=0.0,
    verbose=False,
    leave_progress_bar = True,
    fewshot_frames= 2,
    fewshot_iterations= 5000,
    existed=False):
    if not existed:
      if temp_fewshot_path is None:
          temp_fewshot_path = make_dir("temp_fewshot")


      build_fewshot_directories(temp_fewshot_path)

    train_input_dir = temp_fewshot_path + "/train/input_filtered"
    train_mask_dir = temp_fewshot_path + "/train/mask"
    train_output_dir = temp_fewshot_path + "/train/output"
    gen_input_dir = temp_fewshot_path + "/gen/input_filtered"
    gen_output_dir = temp_fewshot_path + "/gen/output"

    if not existed:

      mypath = video_frames_path
      onlyfiles = list(sorted([f for f in listdir(mypath) if isfile(join(mypath, f))]))

      filenames = []
      d = fewshot_frames
      # iterating each number in list   
      for chunk in np.array_split(onlyfiles, fewshot_frames):
        middle = math.trunc(len(chunk)/2)
        filenames.append(chunk[middle].split(".")[0])
      #match aspect ratio first frame 
      output_size_X, output_size_Y = (eng_config.output_image_size[0], eng_config.output_image_size[1])


      for subdir, dirs, files in os.walk(video_frames_path):
          for count, file in enumerate(files):
              img = cv2.imread(os.path.join(subdir, file)) 
              file = file.split(".")[0]
              img_stretch = cv2.resize(img, (output_size_X, output_size_Y))
              cv2.imwrite(os.path.join(gen_input_dir,file+".png"), img_stretch)
              if file in filenames:
                  cv2.imwrite(os.path.join(train_input_dir,file+".png"), img_stretch)

                  blank_dir = os.path.join(train_mask_dir,file+".png")
                  blank_image = 255 * np.ones(shape=[output_size_Y, output_size_X, 3], dtype=np.uint8)
                  cv2.imwrite(blank_dir, blank_image)

      video_frames = sorted(glob.glob(f'{train_input_dir}{os.sep}*.png'))

      metadata_comment = style_transfer_per_frame(video_frames,
                                          eng_config,
                                          text_prompts,
                                          image_prompts,
                                          noise_prompts,
                                          iterations_per_frame,
                                          current_source_frame_image_weight,
                                          train_output_dir,
                                          current_source_frame_prompt_weight,
                                          verbose,
                                          leave_progress_bar,
                                          output_extension='png')
    

      # resize output
      for filename in glob.iglob(train_output_dir + '**/*.png', recursive=True):
          im = Image.open(filename)
          im = im.resize((output_size_X, output_size_Y), Image.ANTIALIAS)
          im.save(filename , 'png')
    

    st(fewshot_config_file,"logs_reference_P",temp_fewshot_path,fewshot_iterations)

    copy_tree(gen_output_dir, output_path)

    return metadata_comment

    




def build_fewshot_directories(temp_fewshot_path):
    paths = {'train':[
                        'input_filtered',
                        'mask',
                        'output'
                     ],
             'gen':['input_filtered']
            }
    for folder, sub in paths.items():
        for dir in sub:
          make_dir(os.path.join(temp_fewshot_path,os.path.join(folder,dir)))




def make_dir(path):
    if len(path) > 0:
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            for f in os.listdir(path):
                shutil.rmtree(os.path.join(path, f))
        return path

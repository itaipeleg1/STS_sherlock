from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import requests
import os
import copy
import torch
import sys
import warnings
from decord import VideoReader, cpu
import numpy as np

import torch
print(torch.cuda.is_available())
# Open the video and extract frames
video_path = "/home/new_storage/sherlock/STS_sherlock/projects data/Sherlock.S01E01.A.Study.in.Pink.1080p.10bit.BluRay.5.1.x265.HEVC-MZABI.mkv"
video = VideoReader(video_path, ctx=cpu(0), num_threads=1)
start_frame = 11928
end_frame = 11928 + 3*24
vr = video.get_batch(list(range(start_frame, end_frame))).asnumpy()  # convert to numpy array

warnings.filterwarnings("ignore")
def load_video(vr, max_frames_num, fps=1, force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    
    total_frame_num = len(vr)
    video_time = total_frame_num / video.get_avg_fps()  # use original video's fps
    fps = round(video.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/video.get_avg_fps() for i in frame_idx]
    
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr[frame_idx]
    
    return spare_frames, frame_time, video_time

pretrained = "lmms-lab/LLaVA-Video-7B-Qwen2"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(
    pretrained, 
    None, 
    model_name, 
    torch_dtype="bfloat16", 
    device_map=device_map,
    cache_dir="/home/new_storage/sherlock/hf_cache",
    attn_implementation=None
)
model.eval()

max_frames_num = 64
video, frame_time, video_time = load_video(vr, max_frames_num, 1, force_sample=True)
video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().half()
video = [video]

# Rest of the code remains the same...

conv_template = "qwen_1_5"

# Simple timing information
time_instruction = f"The video lasts for {video_time:.2f} seconds with {len(video[0])} sampled frames."

# Change this to your specific question
your_question = "Is there social interaction in this video?"

# Combine with the image token and timing information
question = DEFAULT_IMAGE_TOKEN + f"\n{time_instruction}\n{your_question}"

# Set up the conversation
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

# Generate response
input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
cont = model.generate(
    input_ids,
    images=video,
    modalities=["video"],
    do_sample=False,
    temperature=0,
    max_new_tokens=1, 
)

# Output the result
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
print(f"Question: {your_question}")
print(f"Answer: {text_outputs}")
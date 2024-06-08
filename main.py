from decord import VideoReader, cpu

import torch

import numpy as np

from huggingface_hub import hf_hub_download


from google.colab import drive


# Mount Google Drive
drive.mount('/content/drive')

# Load data
file_path = 'PATH'

np.random.seed(0)

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):

   converted_len = int(clip_len * frame_sample_rate)

   end_idx = np.random.randint(converted_len, seg_len)

   start_idx = end_idx - converted_len

   indices = np.linspace(start_idx, end_idx, num=clip_len)

   indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)

   return indices

videoreader = VideoReader(file_path, num_threads=1, ctx=cpu(0))

# sample 32 frames

videoreader.seek(0)

indices = sample_frame_indices(clip_len=32, frame_sample_rate=4, seg_len=len(videoreader))

video = videoreader.get_batch(indices).asnumpy()

from PIL import Image

Image.fromarray(video[0])


import matplotlib.pyplot as plt

# Visualize 32 frames

fig, axs = plt.subplots(4, 8, figsize=(16, 8))

fig.suptitle('Sampled Frames from Video')

axs = axs.flatten()

for i in range(32):

   axs[i].imshow(video[i])

   axs[i].axis('off')

plt.tight_layout()

plt.savefig('Frames')

plt.show()


from transformers import XCLIPProcessor, XCLIPModel

model_name = "microsoft/xclip-base-patch16-zero-shot"

processor = XCLIPProcessor.from_pretrained(model_name)

model = XCLIPModel.from_pretrained(model_name)

import torch

input_text=[ENTER VIDEO CLASSES]

inputs = processor(text=input_text, videos=list(video), return_tensors="pt", padding=True)

# forward pass

with torch.no_grad():

   outputs = model(**inputs)

probs = outputs.logits_per_video.softmax(dim=1)

probs

max_prob=torch.argmax(probs)

print(f'Video is about :  {input_text[max_prob]}')

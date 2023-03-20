import torch
import torch.nn as nn
from torch.nn import functional as F
from src.open_clip import AudioCLIP, CLIPAudioCfg, CLIPTextCfg
import webdataset as wds
import sys
import os
from torchvision import transforms
from PIL import Image
import numpy as np
import time
import multiprocessing as mp
import wandb
from torch import optim
import sys
import open_clip

device = 'cuda'

params = {
    'lr': 0.001,
    'beta1': 0.99,
    'beta2': 0.999,
    'weight_decay': 0,
    'batch_size': 128,
    'epochs': 10,
    'output_file': 'checkpoints'
}

class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        total_loss = (
                             F.cross_entropy(logits_per_image, labels) +
                             F.cross_entropy(logits_per_text, labels)
                     ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss



transform = transforms.Compose([
    transforms.ToTensor()
])


def preprocess(sample:tuple):
    '''
    Resamples audio to target_sr and selects random seq_len seconds segment from audio
    if audio is shorter than seq_len, repeats audio k times (k=seq_len/audio_len, where audio_len lenght of audio)
    Converts all audio samples to mono format
    Converts captions from JSON to string:
        if there's audio meta tags like title, artist, genre constructs caption playing {genre} song "{title}" by {artist}
        uses raw cpation otherwise
    '''
    image, json_data = sample
    # json_data = json.loads(json_data.decode())
   
    audio_meta = json_data.get('audio_meta', None)
    
    if audio_meta is not None:
        tags = audio_meta.get('tags', None)
        if tags is not None:
            try:
                title, artist, genre = '', '', ''
                for k in tags.keys():
                    if k in ['title', 'TITLE']:
                        title = f'titled {tags[k]}'
                    if k in ['artist', 'ARTIST']:
                        artist = f'by {tags[k]}'
                    if k in ['genre', 'GENRE']:
                        genre = tags[k]

                label = f'{genre} song "{title}" {artist}'
            except:
                pass
    label = f'{json_data["caption"]}'

    return image, label

def get_dataset(urls: list):
    '''
    Pass s3 urls and get processed torch dataset
    '''
    urls = [f'pipe:aws s3 cp {url} -' for url in urls]
    dataset = (
           wds.WebDataset(urls)
           .decode("pil")
           .to_tuple("jpg", "json")
           .map_tuple(transform)
           .map(preprocess)
    )
    return dataset

urls = [f's3://s-laion/CC_AUDIO_spectrograms/{i:05d}.tar' for i in range(10)]
dataset = get_dataset(urls)

batch_size = 32
n_workers = 5
loader = torch.utils.data.DataLoader(dataset, batch_size=params['batch_size'], num_workers=n_workers)

s = time.time()

mulan = AudioCLIP(
    embed_dim = 32,
    audio_cfg = CLIPAudioCfg(**{'image_size': (512, 1001), 'patch_size': 32}),
    text_cfg = CLIPTextCfg(),
    cast_dtype = torch.float16
)

tokenizer = open_clip.get_tokenizer('ViT-B-32-quickgelu')

mulan.to(device)
criterion = ClipLoss()
optimizer = optim.AdamW(mulan.parameters(), lr=params['lr'], betas=(params['beta1'], params['beta1']))

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

os.makedirs(params["output_file"], exist_ok=True)
for epoch in range(params['epochs']):
    for i, batch in enumerate(loader):
        images, labels = batch
        labels = tokenizer(list(labels))
        # images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        mulan.train()
        images = images.to(device)
        audio_latent, text_latent, logit_scale = mulan(images, labels)
        loss = criterion(text_features=text_latent, image_features=audio_latent, logit_scale=logit_scale)
        loss.backward()
        optimizer.step()
    print(f'{epoch+1}/{params["epochs"]} loss: {loss.item()} ')

    scheduler.step()
    torch.save(mulan.state_dict(), f'{params["output_file"]}/model_{epoch}.pth' )

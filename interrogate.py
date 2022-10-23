"""
MIT License

Copyright (c) 2022 pharmapsychotic
https://github.com/pharmapsychotic/clip-interrogator/blob/main/clip_interrogator.ipynb
"""

import numpy as np
import os
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import CLIPTokenizer, CLIPModel
from transformers import CLIPProcessor, CLIPModel

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "blip_model", "data")
def load_list(filename):
    with open(filename, 'r', encoding='utf-8', errors='replace') as f:
        items = [line.strip() for line in f.readlines()]
    return items

artists = load_list(os.path.join(data_path, 'artists.txt'))
flavors = load_list(os.path.join(data_path, 'flavors.txt'))
mediums = load_list(os.path.join(data_path, 'mediums.txt'))
movements = load_list(os.path.join(data_path, 'movements.txt'))

sites = ['Artstation', 'behance', 'cg society', 'cgsociety', 'deviantart', 'dribble', 'flickr', 'instagram', 'pexels', 'pinterest', 'pixabay', 'pixiv', 'polycount', 'reddit', 'shutterstock', 'tumblr', 'unsplash', 'zbrush central']
trending_list = [site for site in sites]
trending_list.extend(["trending on "+site for site in sites])
trending_list.extend(["featured on "+site for site in sites])
trending_list.extend([site+" contest winner" for site in sites])

device="cpu"
blip_image_eval_size = 384
clip_name="openai/clip-vit-large-patch14"

blip_model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_base_caption.pth'  

def generate_caption(blip_model, pil_image, device="cpu"):
    gpu_image = transforms.Compose([
        transforms.Resize((blip_image_eval_size, blip_image_eval_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        caption = blip_model.generate(gpu_image, sample=False, num_beams=3, max_length=20, min_length=5)
    return caption[0]

def rank(text_features, image_features, text_array, top_count=1):
    top_count = min(top_count, len(text_array))
    similarity = torch.zeros((1, len(text_array)))
    for i in range(image_features.shape[0]):
        similarity += (100.0 * image_features[i].unsqueeze(0) @ text_features.T).softmax(dim=-1)
    similarity /= image_features.shape[0]

    top_probs, top_labels = similarity.cpu().topk(top_count, dim=-1)  
    return [(text_array[top_labels[0][i].numpy()], (top_probs[0][i].numpy()*100)) for i in range(top_count)]

class Interrogator:
    def __init__(self) -> None:
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_name)
        try:
            self.get_blip()
        except:
            self.blip_model = None
        self.model = CLIPModel.from_pretrained(clip_name)
        self.processor = CLIPProcessor.from_pretrained(clip_name)
        self.text_feature_lst = [torch.load(os.path.join(data_path, f"{i}.pth")) for i in range(5)]

    def get_blip(self):
        from blip_model.blip import blip_decoder
        blip_model = blip_decoder(pretrained=blip_model_url, image_size=blip_image_eval_size, vit='base')
        blip_model.eval()
        self.blip_model = blip_model


    def interrogate(self,image,use_caption=False):
        if self.blip_model:
            caption = generate_caption(self.blip_model, image)
        else:
            caption = ""
        model,processor=self.model,self.processor
        bests = [[('',0)]]*5
        if True:
            print(f"Interrogating with {clip_name}...")

            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            ranks = [
                rank(self.text_feature_lst[0], image_features, mediums),
                rank(self.text_feature_lst[1], image_features, ["by "+artist for artist in artists]),
                rank(self.text_feature_lst[2], image_features, trending_list),
                rank(self.text_feature_lst[3], image_features, movements),
                rank(self.text_feature_lst[4], image_features, flavors, top_count=3)
            ]

            for i in range(len(ranks)):
                confidence_sum = 0
                for ci in range(len(ranks[i])):
                    confidence_sum += ranks[i][ci][1]
                if confidence_sum > sum(bests[i][t][1] for t in range(len(bests[i]))):
                    bests[i] = ranks[i]

        flaves = ', '.join([f"{x[0]}" for x in bests[4]])
        medium = bests[0][0][0]
        print(ranks)
        if caption.startswith(medium):
            return f"{caption} {bests[1][0][0]}, {bests[2][0][0]}, {bests[3][0][0]}, {flaves}"
        else:
            return f"{caption}, {medium} {bests[1][0][0]}, {bests[2][0][0]}, {bests[3][0][0]}, {flaves}"









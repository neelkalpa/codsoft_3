import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from sys import argv

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large",cache_dir='./models')
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large",cache_dir='./models').to("cuda" if torch.cuda.is_available() else "cpu")

raw_image = Image.open(argv[1])

inputs = processor(raw_image, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

out = model.generate(**inputs,max_new_tokens=200)
print(processor.decode(out[0], skip_special_tokens=True))
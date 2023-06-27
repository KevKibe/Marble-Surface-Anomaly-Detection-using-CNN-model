import io
import requests
from PIL import Image
import torch
import torchvision.transforms as transforms

url = 'http://localhost:5000/predict'
image_path = "Test Images/istockphoto-511661914-612x612.jpg"
pil_image = Image.open(image_path)

transform = transforms.ToTensor()
tensor_image = transform(pil_image)

img_bytes = io.BytesIO()
transforms.ToPILImage()(tensor_image.cpu()).save(img_bytes, format='PNG')

r = requests.post(url, files={'file': ('image.png', img_bytes.getvalue(), 'image/png')})

print(r.text)

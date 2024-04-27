from forest import UNet
import torch
from PIL import Image
import torchvision.transforms.functional as TF

model = UNet(in_channels=3, out_channels=1)

model.load_state_dict(torch.load('model.pth'))

image_name = 'Amazon_5.tiff_50.tiff'
image = Image.open(f'./AmazonForestDataset/Training/images/{image_name}').convert('RGB')
image = TF.to_tensor(image)

prediction = model(image.unsqueeze(0))
prediction = TF.to_pil_image(prediction.squeeze(0))
prediction.save(f'prediction_{image_name}.png')
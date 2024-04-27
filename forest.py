import os
import torch
from torch import nn
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, Compose
import torchvision.transforms.functional as TF

# Constants
BATCH_SIZE = 4
NUM_WORKERS = 0

# Dataset class

transform = Compose([
	Resize((512, 512)),  # Ensure all images are the correct size
	ToTensor()
])

class ForestDataset(Dataset):
	def __init__(self, images_dir, masks_dir, transform=None):
		self.images_dir = images_dir
		self.masks_dir = masks_dir
		self.transform = transform
		self.images = sorted([os.path.join(images_dir, file) for file in os.listdir(images_dir) if file.endswith('.tiff')])
		self.masks = sorted([os.path.join(masks_dir, file) for file in os.listdir(masks_dir) if file.endswith('.png')])

	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):
		image = Image.open(self.images[idx]).convert('RGB')
		mask = Image.open(self.masks[idx]).convert('L')  # Assuming mask is one channel (binary)
		
		if self.transform:
			image = self.transform(image)
			mask = self.transform(mask)

		return image, mask.float()  # Ensure mask is floating-point for loss calculation compatibility


# Use the transform in the dataset instantiation
train_dataset = ForestDataset(
	'AmazonForestDataset/Training/images',
	'AmazonForestDataset/Training/masks',
	transform=transform
)
validation_dataset = ForestDataset(
	'AmazonForestDataset/Training/images',
	'AmazonForestDataset/Training/masks',
	transform=transform
)


# Data Loader
train_dataset_loader = DataLoader(
	train_dataset,
	batch_size=4,
	shuffle=True,
	num_workers=4,
	pin_memory=True,
)

test_dataset_loader = DataLoader(
	validation_dataset,
	batch_size=BATCH_SIZE,
	shuffle=False,
	num_workers=NUM_WORKERS,
	pin_memory=True,
)


# Device
device = (
	"cuda" if torch.cuda.is_available() else
	"mps" if torch.backends.mps.is_available() else
	"cpu"
)
print(f"Using device {device}")


# Model
class DoubleConv(nn.Module):
	"""(convolution => [BN] => ReLU) * 2"""

	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.double_conv = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		print(f"Input shape: {x.shape}")
		return self.double_conv(x)

class NeuralNetwork(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels

		self.encoder1 = DoubleConv(in_channels, 64)
		self.pool1 = nn.MaxPool2d(2)
		self.encoder2 = DoubleConv(64, 128)
		self.pool2 = nn.MaxPool2d(2)
		self.encoder3 = DoubleConv(128, 256)
		self.pool3 = nn.MaxPool2d(2)
		self.encoder4 = DoubleConv(256, 512)
		self.pool4 = nn.MaxPool2d(2)

		self.bottleneck = DoubleConv(512, 1024)

		self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
		self.decoder4 = DoubleConv(1024, 512)
		self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
		self.decoder3 = DoubleConv(512, 256)
		self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
		self.decoder2 = DoubleConv(256, 128)
		self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
		self.decoder1 = DoubleConv(128, 64)

		self.conv_last = nn.Conv2d(64, out_channels, kernel_size=1)

	def forward(self, x):
		# Encoder path
		x1 = self.encoder1(x)
		x2 = self.encoder2(self.pool1(x1))
		x3 = self.encoder3(self.pool2(x2))
		x4 = self.encoder4(self.pool3(x3))

		# Bottleneck
		bottleneck = self.bottleneck(self.pool4(x4))

		# Decoder path
		x = self.upconv4(bottleneck)
		x = self.decoder4(torch.cat((x, x4), dim=1))
		x = self.upconv3(x)
		x = self.decoder3(torch.cat((x, x3), dim=1))
		x = self.upconv2(x)
		x = self.decoder2(torch.cat((x, x2), dim=1))
		x = self.upconv1(x)
		x = self.decoder1(torch.cat((x, x1), dim=1))

		return self.conv_last(x)

# Example of creating a NeuralNetwork for 3-channel input images and 1-channel output masks
model = NeuralNetwork(in_channels=3, out_channels=1)


# Loss function
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# Training function
def train(dataloader, model, loss_fn, optimizer, epoch):
	model.train()
	for batch, (X, y) in enumerate(dataloader):
		print(f"Batch {batch}")
		X, y = X.to(device), y.to(device)
		optimizer.zero_grad()
		pred = model(X)
		loss = loss_fn(pred, y)
		print(f"Loss: {loss.item()}")
		loss.backward()
		optimizer.step()
		if batch % 10 == 0:  # Adjust print frequency as needed
			print(f"Batch {batch}, Loss: {loss.item()}")


# Single Test Loop
def test(dataloader, model, loss_fn):
	size = len(dataloader.dataset)
	num_batches = len(dataloader)
	model.eval()
	test_loss, correct = 0, 0
	with torch.no_grad():
		for X, y in dataloader:
			X, y = X.to(device), y.to(device)
			pred = model(X)
			test_loss += loss_fn(pred, y).item()
			correct += (pred.argmax(1) == y).type(torch.float).sum().item()
	test_loss /= num_batches
	correct /= size
	print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def printimage(model, image_name, epoch):
	image = Image.open(f'./AmazonForestDataset/Training/images/{image_name}').convert('RGB')
	image = TF.to_tensor(image)

	prediction = model(image.unsqueeze(0))
	prediction = TF.to_pil_image(prediction.squeeze(0))
	prediction.save(f'exports/{image_name}_epoch{epoch}.png')

def main():
	model.load_state_dict(torch.load('model.pth'))

	# Training loop
	print("Starting Training...")
	for epoch in range(5):
		epoch += 5
		print(f"Training Epoch {epoch}\n-------------------------------")
		train(train_dataset_loader, model, loss_fn, optimizer, epoch)
		
		# Save the model
		torch.save(model.state_dict(), f"model{epoch}.pth")
		# model.load_state_dict(torch.load('model.pth'))

		print(f"Printing images for epoch {epoch}")
		printimage(model, "Amazon_5.tiff_50.tiff", epoch)
		printimage(model, "Amazon_122.tiff_33.tiff", epoch)

		#test(test_dataset_loader, model, loss_fn)

	print("endjawoll")

if __name__ == "__main__":
	main()
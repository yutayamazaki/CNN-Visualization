IMG_DIR = '../img/'
SAMPLE_IMG = 'cat.jpg'

IMG_SIZE = 224

from torchvision import transforms

transform = transforms.Compose

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])
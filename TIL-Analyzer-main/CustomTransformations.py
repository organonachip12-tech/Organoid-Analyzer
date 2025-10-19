import torch
from PIL import Image, ImageDraw
from torchvision import datasets, transforms
import random

seed = 1
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

class CenterCircleCrop(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        size = img.size

        mask = Image.new('L', size, 0)
        draw = ImageDraw.Draw(mask) 
        draw.ellipse((0,0) + size, fill=255)
        output = Image.new('RGB', img.size, (0, 0, 0))
        output.paste(img, (0, 0), mask)

        return output

class CenterCircleCrop2(torch.nn.Module):
    def __init__(self, radius):
        super().__init__()
        self.radius = radius

    def forward(self, img):
        size = img.size


        circle_center_x = size[0] // 2
        circle_center_y = size[1] // 2

        mask = Image.new('L', size, 0)
        draw = ImageDraw.Draw(mask) 
        draw.ellipse((circle_center_x - self.radius, circle_center_y - self.radius,  
                      circle_center_x + self.radius, circle_center_y + self.radius), fill=255)
        output = Image.new('RGB', img.size, (0, 0, 0))
        output.paste(img, (0, 0), mask)

        return output

class ImageTransformations():
    #640, 40, 4
    #tried 512
    #544
    def __init__(self, original_image_size = 640, cropped_image_size = 512):
        self.original_image_size = original_image_size
        self.cropped_image_size = cropped_image_size
        self.using_cropped_size_directly = True

        
        # Transformations
        if not self.using_cropped_size_directly:

            self.final_image_size = self.original_image_size

            self.train_transformations = transforms.Compose([
                transforms.Resize((self.original_image_size, self.original_image_size)),
                transforms.RandomCrop((self.cropped_image_size,self.cropped_image_size)),
                CenterCircleCrop(),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomRotation(degrees=(0, 360)),
                transforms.Resize((self.original_image_size, self.original_image_size)),
                #transforms.Pad(scale_factor//2*scale),
                #transforms.ColorJitter(brightness=.5, contrast=.5, saturation=0.5, hue=0.5),
                
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
            self.validation_transformations = transforms.Compose([
                transforms.Resize((self.original_image_size, self.original_image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])    

            
        else:
            self.final_image_size = self.cropped_image_size

            self.train_transformations = transforms.Compose([
                transforms.Resize((self.original_image_size, self.original_image_size)),
                transforms.RandomCrop((self.cropped_image_size,self.cropped_image_size)),
                CenterCircleCrop(),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomRotation(degrees=(0, 360)),
                #transforms.ColorJitter(brightness=.5, contrast=.5, saturation=0.5),
                
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

            self.validation_transformations = transforms.Compose([
                transforms.Resize((self.cropped_image_size, self.cropped_image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        self.umap_transformations = transforms.Compose([
                transforms.Resize((self.original_image_size, self.original_image_size)),
                transforms.CenterCrop((self.cropped_image_size,self.cropped_image_size)),
                #CenterCircleCrop(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
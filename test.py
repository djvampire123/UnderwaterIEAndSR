import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import sys
import numpy as np
from model import UnifiedEnhanceSuperResNet  # Corrected import

def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    if transform:
        image = transform(image).unsqueeze(0)
    return image

def save_image(tensor, output_path):
    image = tensor.squeeze(0).cpu().detach().numpy()
    image = np.transpose(image, (1, 2, 0))  # CHW to HWC
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    image = Image.fromarray(image)
    image.save(output_path)

def test_model(model, test_images_dir, output_dir):
    # Define the transformation for input images
    transform = transforms.Compose([
        transforms.Resize((240, 320)),  # Resize to match the model input size
        transforms.ToTensor()
    ])

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each image in the test directory
    for image_name in os.listdir(test_images_dir):
        image_path = os.path.join(test_images_dir, image_name)
        output_path = os.path.join(output_dir, image_name)

        # Load and transform the image
        input_image = load_image(image_path, transform)
        input_image = input_image.cuda()

        # Run the model
        with torch.no_grad():
            output_image = model(input_image)

        # Save the output image
        save_image(output_image, output_path)
        print(f"Processed and saved: {output_path}")

def main():
    test_images_dir = sys.argv[1]  # Directory containing test images
    output_dir = sys.argv[2]       # Directory to save output images
    model_checkpoint = sys.argv[3] # Path to the model checkpoint

    # Load the model
    model = UnifiedEnhanceSuperResNet(upscale_factor=2)
    model = nn.DataParallel(model)
    model = model.cuda()

    # Load the trained model weights
    checkpoint = torch.load(model_checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    # Test the model
    test_model(model, test_images_dir, output_dir)

if __name__ == '__main__':
    main()
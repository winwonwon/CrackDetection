import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torchvision.transforms import functional as F
import os
import argparse

from model import Model1, Model2, pretrainedModel  # ensure correct names
from dl_utils import get_transform

def load_model(model_path, model_class, num_classes=2, device='cuda'):
    """Load trained model from checkpoint"""
    model = model_class(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    model.device = device  # Attach for later use
    return model

def predict_and_visualize(image_path, model, transform, threshold=0.5, save_dir=None):
    """Make prediction and visualize results for a single image"""
    image = Image.open(image_path).convert("RGB")
    transformed = transform(image)
    img_tensor = transformed.to(model.device)

    with torch.no_grad():
        prediction = model([img_tensor])[0]

    # Denormalize for visualization
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    denorm = img_tensor.cpu() * std + mean
    denorm = torch.clamp(denorm, 0, 1)
    img_pil = F.to_pil_image(denorm)

    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(img_pil)

    for box, score in zip(prediction['boxes'], prediction['scores']):
        if score >= threshold:
            xmin, ymin, xmax, ymax = box.cpu().numpy()
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                     linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(xmin, ymin - 5, f"{score:.2f}", color="red", fontsize=8)

    plt.title(f"Predictions (threshold ≥ {threshold})")
    plt.axis('off')

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        output_path = os.path.join(save_dir, os.path.basename(image_path))
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def process_folder(model, transform, input_folder="test_images", output_folder="output_predictions", threshold=0.5):    
    """Process all images in a folder"""
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(valid_extensions):
            image_path = os.path.join(input_folder, filename)
            predict_and_visualize(image_path, model, transform, threshold, output_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Prediction Script')
    parser.add_argument('--model_type', type=str, choices=['1', '2', 'p'], default='2',
                        help='Model selection: 1=Model1, 2=Model2, p=Pretrained')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Confidence threshold for predictions (0-1)')
    args = parser.parse_args()

    model_config = {
        '1': {'class': Model1, 'path': 'model_Model1.pth'},
        '2': {'class': Model2, 'path': 'model_Model2.pth'},
        'p': {'class': pretrainedModel, 'path': 'model_PreTrained.pth'}
    }

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using CUDA" if torch.cuda.is_available() else "Using CPU")

    try:
        selected = model_config[args.model_type]
        model = load_model(selected['path'], selected['class'], num_classes=2, device=device)

        process_folder(
            model=model,
            transform=get_transform(train=False),
            threshold=args.threshold
        )

    except KeyError:
        print(f"Error: Invalid model type '{args.model_type}'. Valid options: 1, 2, p")
    except FileNotFoundError as e:
        print(f"Model file not found: {e}")

# python predict.py --model_type 1 --threshold 0.5
# python predict.py --model_type 1 --threshold 0.1
# python predict.py --model_type 2 --threshold 0.5
# python predict.py --model_type 2 --threshold 0.1
# python predict.py --model_type p --threshold 0.5
# python predict.py --model_type p --threshold 0.8
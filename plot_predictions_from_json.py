import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from torchvision.transforms import functional as F
import os

from dataset import CocoCrackDataset
from model import pretrainedModel, Model1, Model2
from dl_utils import get_transform

def visualize_predictions(model, dataset, device, num_images=8, threshold=0.01, output_dir="viz_results"):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    # Define normalization parameters (should match your training params)
    mean = torch.tensor([0.5,0.5,0.5]).view(3, 1, 1)
    std = torch.tensor([0.5,0.5,0.5]).view(3, 1, 1)
    # mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    # std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for idx in range(num_images):
        img, target = dataset[idx]
        img_tensor = img.to(device)

        with torch.no_grad():
            prediction = model([img_tensor])[0]

        # Reverse normalization and convert to PIL
        denormalized = img.cpu() * std + mean 
        denormalized = torch.clamp(denormalized, 0, 1) 
        img_np = F.to_pil_image(denormalized)

        fig, ax = plt.subplots(1)
        ax.imshow(img_np)

        for box in target['boxes']:
            xmin, ymin, xmax, ymax = box.cpu()
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    linewidth=2, edgecolor='green', facecolor='none')
            ax.add_patch(rect)

        for box, score in zip(prediction['boxes'], prediction['scores']):
            if score >= threshold:
                xmin, ymin, xmax, ymax = box.cpu()
                rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                        linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                ax.text(xmin, ymin - 5, f"{score:.2f}", color="red", fontsize=8)

        plt.axis('off')
        plt.title(f"Image {idx+1}: Green=GT, Red=Pred ≥ {threshold}")
        save_path = os.path.join(output_dir, f"prediction_{idx+1:03d}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = CocoCrackDataset(
        root="dataset/test/images",
        annotation="dataset/test/_annotations.coco.json",
        transforms=get_transform(train=False)
    )

    # Load model
    model = Model2(num_classes=2)
    model.load_state_dict(torch.load("model_Model2.pth", map_location=device))
    model.to(device)

    visualize_predictions(model, dataset, device, num_images=40, threshold=0.5, output_dir="viz_results")

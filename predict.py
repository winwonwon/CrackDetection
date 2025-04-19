import torch
from dataset import CocoCrackDataset
from model import get_model_instance
from dl_utils import get_transform
import os
from PIL import ImageDraw
import torchvision.transforms.functional as F

def load_test_dataset(data_dir):
    return CocoCrackDataset(
        root=os.path.join(data_dir, "test/images"),
        annotation=os.path.join(data_dir, "test/_annotations.coco.json"),
        transforms=get_transform(train=False)
    )

def predict_and_save(model, dataset, device, output_dir="predictions", threshold=0.5):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    for idx in range(len(dataset)):
        image, target = dataset[idx]
        image_tensor = image.to(device)

        with torch.no_grad():
            prediction = model([image_tensor])[0]

        img_pil = F.to_pil_image(image_tensor.cpu())
        draw = ImageDraw.Draw(img_pil)

        for box, score in zip(prediction['boxes'], prediction['scores']):
            if score >= threshold:
                xmin, ymin, xmax, ymax = box
                draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)

        img_name = f"image_{idx:03d}.jpg"
        img_pil.save(os.path.join(output_dir, img_name))
        print(f"Saved: {img_name}")

if __name__ == "__main__":
    data_path = "dataset"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model_instance(num_classes=2)
    model.load_state_dict(torch.load("crack_model.pth"))  # <-- save your trained model after training
    model.to(device)

    test_ds = load_test_dataset(data_path)
    predict_and_save(model, test_ds, device)

import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import CocoCrackDataset
from torch.utils.tensorboard import SummaryWriter
from dl_utils import get_transform, collate_fn
from model import pretrainedModel, Model1, Model2, Model6
from evaluate import evaluate_model
from visualize import visualize_predictions
from torch.amp import GradScaler, autocast
import os

def load_datasets(data_dir):
    train_ds = CocoCrackDataset(
        root=os.path.join(data_dir, "train/images"),
        annotation=os.path.join(data_dir, "train/_annotations.coco.json"),
        transforms=get_transform(train=True)
    )
    val_ds = CocoCrackDataset(
        root=os.path.join(data_dir, "valid/images"),
        annotation=os.path.join(data_dir, "valid/_annotations.coco.json"),
        transforms=get_transform(train=False)
    )
    test_ds = CocoCrackDataset(
        root=os.path.join(data_dir, "test/images"),
        annotation=os.path.join(data_dir, "test/_annotations.coco.json"),
        transforms=get_transform(train=False)
    )
    return train_ds, val_ds, test_ds

def train_model(data_dir, num_epochs=10, batch_size=4, model_choice="Model6"):
    log_dir = os.path.join("runs", model_choice)
    writer = SummaryWriter(log_dir=log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, val_ds, test_ds = load_datasets(data_dir)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                            collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                           collate_fn=collate_fn, num_workers=4, pin_memory=True)

    # Model selection dictionary
    model_dict = {
        "PreTrained": pretrainedModel,
        "Model1": Model1,
        "Model2": Model2,
        "Model6": Model6
    }

    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")  # Fixed syntax
    
    # Initialize model
    model = model_dict[model_choice](num_classes=2)

    # Freeze backbone layers to speed up training
    # for name, param in model.backbone.named_parameters():
    #     param.requires_grad = False
    
    model.to(device)
    model.eval()

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
        lr=0.0005,
        weight_decay=0.01
    )
    
    # Initialize GradScaler with new syntax
    scaler = GradScaler(device='cuda')

    print(f"\nTraining {model_choice} for {num_epochs} epochs...\n")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        start_time = time.time()

        for step, (images, targets) in enumerate(train_loader):
            images = [img.to(device, non_blocking=True) for img in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

            # Mixed precision forward pass
            with autocast(device_type='cuda'):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            # Scaled backward pass
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(losses).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += losses.item()
            writer.add_scalar('Loss/train_step', losses.item(), epoch * len(train_loader) + step)

        elapsed = time.time() - start_time

        writer.add_scalar('Loss/train_epoch', epoch_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        print(f"[Epoch {epoch+1}] Loss: {epoch_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f} | Time: {elapsed:.2f}s")

    # Evaluation and saving
    print("\nEvaluating on validation set...")
    val_ann_file = os.path.join(data_dir, "valid/_annotations.coco.json")
    os.makedirs("outputs/val", exist_ok=True)
    evaluate_model(model, val_ds, val_ann_file, device, results_file="outputs/val/results.json")
    visualize_predictions(model, val_ds, device, output_dir="outputs/val/visualizations")

    print("\nEvaluating on test set...")
    test_ann_file = os.path.join(data_dir, "test/_annotations.coco.json")
    os.makedirs("outputs/test", exist_ok=True)
    evaluate_model(model, test_ds, test_ann_file, device, results_file="outputs/test/results.json")
    visualize_predictions(model, test_ds, device, output_dir="outputs/test/visualizations")

    model_file = f"model_{model_choice}.pth"
    torch.save(model.state_dict(), model_file)
    print(f"\nModel saved to {model_file}")
    writer.close()

if __name__ == "__main__":
    data_path = "dataset"
    train_model(data_path, num_epochs=15, batch_size=16, model_choice="Model2")
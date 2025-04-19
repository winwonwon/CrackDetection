from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import torch

SCORE_THRESHOLD = 0.001

def evaluate_model(model, dataset, annotation_file, device, results_file='results.json', quiet=False):
    model.eval()
    coco_gt = COCO(annotation_file)
    results = []

    for idx in range(len(dataset)):
        img, target = dataset[idx]
        img_id = target['image_id'].item()
        img_tensor = img.to(device)

        with torch.no_grad():
            prediction = model([img_tensor])[0]

        for box, score, label in zip(prediction['boxes'], prediction['scores'], prediction['labels']):
            if score >= SCORE_THRESHOLD:
                xmin, ymin, xmax, ymax = box
                width = xmax - xmin
                height = ymax - ymin
                results.append({
                    'image_id': img_id,
                    'category_id': int(label),
                    'bbox': [xmin.item(), ymin.item(), width.item(), height.item()],
                    'score': score.item()
                })

    with open(results_file, 'w') as f:
        json.dump(results, f)

    if not quiet:
        print(f"Saved predictions to: {results_file}")

    coco_dt = coco_gt.loadRes(results_file)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


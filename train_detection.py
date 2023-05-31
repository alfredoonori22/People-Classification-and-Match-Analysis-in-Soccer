import cv2
import numpy as np
import torch
import torchvision
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.transforms import ToPILImage

from dataset import CLASS_DICT


def draw_bbox(image, target, output):
    image = image.cpu()
    image = ToPILImage()(image)
    image = np.array(image)
    keys = list(CLASS_DICT.keys())

    for i, (x1, y1, x2, y2) in enumerate(output['boxes']):
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        index = (list(CLASS_DICT.values()).index(int(output['labels'][i])))
        cv2.putText(image, keys[index], (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    for (x1, y1, x2, y2) in target['boxes']:
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    cv2.imwrite(f"image/bbox-{target['image_id']}.png", image)


def apply_nms(orig_prediction, iou_thresh=0.3):
    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]

    return final_prediction


def train_one_epoch_detection(model, optimizer, training_loader, epoch):
    running_loss = 0.0
    last_loss = 0.0

    # Make sure gradient tracking is on, and do a pass over the data
    model.train()

    # enumerate(tqdm(training_loader, file=sys.stdout))            enumerate(training_loader)
    for i, (images, targets) in enumerate(training_loader):
        # Every data instance is an image + target pair
        images = list(image.cuda() for image in images)
        targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

        # Make predictions for this batch
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Zero gradients for every batch
        optimizer.zero_grad()
        # Compute the gradients w.r.t. losses
        losses.backward()
        # Adjust learning weights
        optimizer.step()

        # Gather data
        running_loss += losses.item()

        # Print loss every 1000 batches
        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            print(f'  batch {i + 1} loss: {last_loss}')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': running_loss,
            }, "model/checkpoint_detection")
            running_loss = 0.0

    return last_loss


def evaluate(model, validation_loader):
    model.eval()

    with torch.inference_mode():
        # Validation metric
        metric = MeanAveragePrecision(class_metrics=True)

        for i, (images, targets) in enumerate(validation_loader):
            # Singol batch's score
            images = list(image.cuda() for image in images)

            # Predict the output
            outputs = model(images)
            outputs = [{k: v.cpu() for k, v in t.items()} for t in outputs]

            # Non Max Suppression to discard intersected superflous bboxes
            outputs = [apply_nms(o, iou_thresh=0.2) for o in outputs]

            # Keeping only the boxes with high score
            outputs = [
                {'boxes': torch.stack([box for box, score in zip(d['boxes'], d['scores']) if score > 0.3]),
                 'labels': torch.tensor([label for label, score in zip(d['labels'], d['scores']) if score > 0.3]),
                 'scores': torch.tensor([score for score in d['scores'] if score > 0.3])}
                for d in outputs]

            # for k, _ in enumerate(outputs):
                # draw_bbox(images[k], targets[k], outputs[k])

            metric.update(outputs, targets)

        res = metric.compute()
        print(res)

    return res['map']

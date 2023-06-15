import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torchmetrics import F1Score
import transform as T
from utils import apply_nms, ourMetric, draw_bbox


def train_one_epoch_fasterrcnn(model, optimizer, training_loader, epoch, folder):
    running_loss = 0.0
    last_loss = 0.0

    # Make sure gradient tracking is on, and do a pass over the data
    model.train()

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
                'optimizer_state_dict': optimizer.state_dict()
            }, f"{folder}/checkpoint_detection")
            running_loss = 0.0

    return last_loss


def train_one_epoch_cnn(model, optimizer, training_loader, epoch, folder):
    running_loss = 0.0
    last_loss = 0.0

    # Make sure gradient tracking is on, and do a pass over the data
    model.train()

    for i, (images, targets) in enumerate(training_loader):
        # Every data instance is an image + target pair
        images = torch.stack([image.cuda() for image in images])
        targets = torch.stack(targets)
        breakpoint()
        # Make predictions for this batch
        outputs = model(images)
        outputs = outputs.cpu()

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Compute the loss and its gradients
        loss = CrossEntropyLoss()(outputs, targets)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data
        running_loss += loss.item()

        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            print(f'  batch {i + 1} loss: {last_loss}')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, f"{folder}/checkpoint_detection")
            running_loss = 0.0

    return last_loss


def evaluate_fasterrcnn(model, validation_loader):
    model.eval()

    with torch.inference_mode():
        # Validation metric
        count = 0
        mAP_tot = 0
        for count, (images, targets) in enumerate(validation_loader):
            # Singol batch's score
            images = list(image.cuda() for image in images)

            # Predict the output
            outputs = model(images[0].unsqueeze(0))
            outputs = [{k: v.cpu() for k, v in t.items()} for t in outputs]

            # Non Max Suppression to discard intersected superflous bboxes
            outputs = [apply_nms(o, iou_thresh=0.2, thresh=0.7) for o in outputs]

            mAP_tot += ourMetric(outputs, targets)

        mAP_tot /= (count + 1)

    return mAP_tot


def evaluate_cnn(model, validation_loader):
    model.eval()
    metric = F1Score(num_classes=3, average=None, task="multiclass")

    with torch.inference_mode():
        for i, (images, targets) in enumerate(validation_loader):
            # Singol batch's score
            images = torch.stack([image.cuda() for image in images])
            targets = torch.stack(targets)

            # Predict the output
            outputs = model(images)
            _, pred = torch.max(torch.exp(outputs.cpu()), 1)
            metric.update(pred, targets)

        res = metric.compute()

    return res


def test_cnn(fasterrcnn, cnn, test_loader):
    fasterrcnn.eval()
    cnn.eval()

    with torch.inference_mode():
        count = 0
        mAP_tot = 0
        for count, (image, target) in enumerate(test_loader):
            image = image[0].cuda()

            # Predict the output
            output = fasterrcnn(image.unsqueeze(0))
            output = [{k: v.cpu() for k, v in t.items()} for t in output]

            # Non Max Suppression to discard intersected superflous bboxes
            output = [apply_nms(o, iou_thresh=0.2, thresh=0.7) for o in output][0]

            for b, box in enumerate(output['boxes']):
                if output['labels'][b] == 2:
                    box_img = image[:, int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                    box_img, _ = T.ResizeImg()(box_img, target, (40, 80))

                    prediction = cnn(box_img.unsqueeze(0))
                    prediction = nn.Softmax(dim=1)(prediction)

                    _, pred = torch.max(torch.exp(prediction.cpu()), 1)

                    if prediction[0][pred] > 0.7:
                        output['labels'][b] = pred + 2

            draw_bbox(image, target[0], output)
            mAP_tot += ourMetric([output], [target[0]])

        mAP_tot /= (count + 1)

    return mAP_tot

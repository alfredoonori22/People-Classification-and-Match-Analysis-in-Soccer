import cv2
import matplotlib.pyplot as plt
import torch
from torch.nn import CrossEntropyLoss
from torchmetrics import F1Score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
from utils import apply_nms, ourMetric


def train_one_epoch_fasterrcnn(model, optimizer, training_loader, epoch, folder):
    running_loss = 0.0
    last_loss = 0.0

    # Gradient trackin is on
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

        # Print loss every 1000 batches and save checkpoint
        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            print(f'  batch {i + 1} loss: {last_loss}')
            running_loss = 0.0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, f"{folder}/checkpoint_detection")

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

        # Print loss every 1000 batches and save checkpoint
        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            print(f'  batch {i + 1} loss: {last_loss}')
            running_loss = 0.0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, f"{folder}/checkpoint_detection")

    return last_loss


def evaluate_fasterrcnn(model, validation_loader):
    model.eval()

    with torch.inference_mode():
        # Validation metric
        count = 0
        mAP_tot = 0
        for count, (image, target) in enumerate(validation_loader):
            image = image[0].cuda()

            # Predict the output
            output = model(image.unsqueeze(0))
            output = [{k: v.cpu() for k, v in t.items()} for t in output]

            # Non Max Suppression to discard intersected superflous bboxes
            output = apply_nms(output[0], iou_thresh=0.2, thresh=0.7)

            mAP_tot += ourMetric([output], [target[0]])

        mAP_tot /= (count + 1)

    return mAP_tot


def evaluate_cnn(model, validation_loader):
    model.eval()
    metric = F1Score(num_classes=3, average="weighted", task="multiclass")
    pred_tot = []
    target_tot = []

    with torch.inference_mode():
        for i, (images, targets) in enumerate(validation_loader):
            # Singol batch's score
            images = torch.stack([image.cuda() for image in images])
            targets = torch.stack(targets)

            # Predict the output
            outputs = model(images)
            _, pred = torch.max(torch.exp(outputs.cpu()), 1)
            metric.update(pred, targets)

            # Lists for confusion matrix
            pred_tot.extend(pred.tolist())
            target_tot.extend(targets.tolist())

        # Confusion Matrix plot
        matrix = confusion_matrix(target_tot, pred_tot)
        mat = pd.DataFrame(matrix, index=['Player', 'Goalkeeper', 'Referee'], columns=['Player', 'Goalkeeper', 'Referee'])

        plt.subplots()
        ax = sns.heatmap(mat, annot=True, cmap='Oranges', fmt='g')
        ax.set_title("Confusion Matrix")
        plt.savefig("Confusion Matrix CNN.png", dpi=300)

        res = metric.compute()

    return res

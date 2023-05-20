import math
import torch


def train_one_epoch_detection(model, optimizer, training_loader, args):
    running_loss = 0.0

    # Make sure gradient tracking is on, and do a pass over the data
    model.train()

    for i, (images, targets) in enumerate(training_loader):
        # Every data instance is an image + target pair
        images = list(image.cuda() for image in images)
        targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

        # Make predictions for this batch
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # TODO: Capire perchè losses ogni tanto esplode (nan) e poi rimane nan
        if not math.isfinite(losses):
            print(f"Loss is {losses}, the Image ID was {targets[0]['image_id']}")
            continue

        # Zero gradients for every batch!
        optimizer.zero_grad()

        losses.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data
        running_loss += losses.item()

        if i % 1 == 0:
            last_loss = running_loss / 1  # loss per batch
            print(f'  batch {i + 1} loss: {last_loss}')
            running_loss = 0.0


def validation(model, validation_loader):
    model.eval()
    with torch.inference_mode():
        score = 0.0
        for i, (images, targets) in enumerate(validation_loader):
            images = list(image.cuda() for image in images)

            outputs = model(images)
            outputs = [{k: v.cpu() for k, v in t.items()} for t in outputs]

            k = 0
            for k, _ in enumerate(outputs):
                score = score + outputs[k]["scores"].mean()
            score = score / k

        return score

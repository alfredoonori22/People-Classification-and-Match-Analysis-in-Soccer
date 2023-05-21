import math
import sys
import torch
from tqdm import tqdm


def train_one_epoch_detection(model, optimizer, training_loader, args):
    running_loss = 0.0

    # Make sure gradient tracking is on, and do a pass over the data
    model.train()

    for i, (images, targets) in enumerate(tqdm(training_loader, file=sys.stdout)):
        # Every data instance is an image + target pair
        images = list(image.cuda() for image in images)
        targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

        # Make predictions for this batch
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # TODO: Capire perchè losses ogni tanto esplode (nan) e poi rimane nan, sembrerebbe risolto con il gradient clip
        if not math.isfinite(losses):
            tqdm.write(f"Loss is {losses}, the Image ID was {targets[0]['image_id']}")
            continue

        # Zero gradients for every batch
        optimizer.zero_grad()
        # Compute the gradients w.r.t. losses
        losses.backward()
        # Clip the gradients norm to avoid exploding gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
        # Adjust learning weights
        optimizer.step()

        # Gather data
        running_loss += losses.item()

        # Print loss every 1000 batches
        if i % 100 == 99:
            last_loss = running_loss / 100  # loss per batch
            tqdm.write(f'  batch {i + 1} loss: {last_loss}')
            running_loss = 0.0


def validation(model, validation_loader, args):
    model.eval()
    with torch.inference_mode():
        # Global validation score
        score = 0.0
        # Number of batches in validation set
        i = 0
        for i, (images, targets) in enumerate(tqdm(validation_loader, file=sys.stdout)):
            # Singol batch's score
            s_batch = 0.0
            images = list(image.cuda() for image in images)

            # Predict the output
            outputs = model(images)
            outputs = [{k: v.cpu() for k, v in t.items()} for t in outputs]

            # TODO: Si può probabilmente fare meglio
            for k, _ in enumerate(outputs):
                s_img = 0.0
                if outputs[k]["scores"].numel() != 0:
                    s_img = outputs[k]["scores"].mean()
                else:
                    # TODO: Capire peché (solo in alcune epoche) in validation inizia a skippare un sacco di img
                    tqdm.write("skippata")

                s_batch = s_batch + s_img

            score = score + s_batch / args.batch_size

        return score / (i + 1)

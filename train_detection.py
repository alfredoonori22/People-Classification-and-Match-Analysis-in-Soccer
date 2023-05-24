import torch
import torchvision
from tqdm import tqdm


def apply_nms(orig_prediction, iou_thresh=0.3):
    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]

    return final_prediction


def train_one_epoch_detection(model, optimizer, training_loader, epoch, args):
    running_loss = 0.0

    # Make sure gradient tracking is on, and do a pass over the data
    model.train()

    for i, (images, targets) in enumerate(tqdm(training_loader)):
        # Every data instance is an image + target pair
        images = list(image.cuda() for image in images)
        targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

        # Make predictions for this batch
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        images = list(image.cpu() for image in images)
        targets = [{k: v.cpu() for k, v in t.items()} for t in targets]

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

        # Print loss every 100 batches
        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            tqdm.write(f'  batch {i + 1} loss: {last_loss}')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': running_loss,
            }, "model/checkpoint_detection.pt")
            running_loss = 0.0


def validation(model, validation_loader, args):
    model.eval()
    with torch.inference_mode():
        # Global validation score
        score = 0.0
        # Number of batches in validation set
        i = 0
        for i, (images, targets) in enumerate(tqdm(validation_loader)):
            # Singol batch's score
            s_batch = 0.0
            images = list(image.cuda() for image in images)

            # Predict the output
            outputs = model(images)
            images = list(image.cpu() for image in images)
            outputs = [{k: v.cpu() for k, v in t.items()} for t in outputs]

            # Non Max Suppression to discard intersected superflous bboxes
            outputs = [apply_nms(o, iou_thresh=0.2) for o in outputs]


            # TODO: Si può probabilmente fare meglio
            for k, _ in enumerate(outputs):
                s_img = 0.0

                if outputs[k]["scores"].numel() != 0:
                    s_img = outputs[k]["scores"].mean()
                # else:
                    # TODO: Capire peché (solo in alcune epoche) in validation inizia a skippare un sacco di img
                    # tqdm.write("skippata")

                # Stampa di debug
                # print(outputs[k]["scores"].numel())

                s_batch = s_batch + s_img

            score = score + s_batch / args.batch_size

        return score / (i + 1)

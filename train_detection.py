import cv2
import torch
from torchvision.transforms import functional
from torch import nn
from torch.nn import CrossEntropyLoss
from torchmetrics import F1Score
import transform as T
from utils import apply_nms, ourMetric, draw_bbox


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
            draw_bbox(image, output, classes=3)

        mAP_tot /= (count + 1)

    return mAP_tot


def evaluate_cnn(model, validation_loader):
    model.eval()
    metric = F1Score(num_classes=3, average="weighted", task="multiclass")

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


def test_fasterrcnn(model):
    model.eval()

    # Load input video
    video = cv2.VideoCapture('/mnt/beegfs/work/cvcs_2022_group20/test.mp4')
    success, cv_image = video.read()
    # Create output video
    out = cv2.VideoWriter('output_detection.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15,
                          (cv_image.shape[1], cv_image.shape[0]))

    while True:
        # Read frame from video
        success, cv_image = video.read()
        if not success:
            break

        # Image pre-processing
        image = functional.to_tensor(cv_image)
        image = image.cuda()

        # Finding boxes in image
        with torch.inference_mode():
            output = model([image])

        output = {k: v.cpu() for k, v in output[0].items()}
        # Non Max Suppression to discard intersected superflous bboxes
        output = apply_nms(output, iou_thresh=0.2, thresh=0.80)

        image = draw_bbox(image, output, classes=3)
        out.write(image)

    # Release resources
    video.release()
    out.release()
    cv2.destroyAllWindows()


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
            output = apply_nms(output[0], iou_thresh=0.2, thresh=0.7)

            # Loop through all the bboxes found by the model
            for b, box in enumerate(output['boxes']):
                # If model predicted this to be a person, run the cnn to infer his class
                if output['labels'][b] == 2:
                    # Crop the image on the bbox
                    box_img = image[:, int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                    box_img, _ = T.ResizeImg()(box_img, target, (40, 80))

                    prediction = cnn(box_img.unsqueeze(0))

                    # Prediction scores for all the classes
                    prediction = nn.Softmax(dim=1)(prediction)

                    # Predicted class
                    _, pred = torch.max(torch.exp(prediction.cpu()), 1)

                    # Change label in output dictionary
                    if prediction[0][pred] > 0.7:
                        output['labels'][b] = pred + 2      # Key values in MULTI_CLASS_DICT = Key values in PEOPLE_DICT

            # Draw bboxes and save the image
            test_image = draw_bbox(image, output, classes=5)
            cv2.imwrite(f"test-images/bboxes-{target[0]['image_id']}.png", test_image[:, :, ::-1])  # Revert channels order

            mAP_tot += ourMetric([output], [target[0]])

        mAP_tot /= (count + 1)

    return mAP_tot

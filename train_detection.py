import torch.nn
from tqdm import tqdm


def train_one_epoch_detection(model, optimizer, training_loader, args):
    epochs = args.epochs
    batch_size = args.batch_size
    running_loss = 0.0
    last_loss = 0.0

    # Defining Loss
    loss_fn = torch.nn.CrossEntropyLoss()

    # Make sure gradient tracking is on, and do a pass over the data
    model.train()

    for i, (images, targets) in enumerate(training_loader):

        # Every data instance is an image + target pair
        images = list(image.cuda() for image in images)
        targets = [{k: v.squeeze().cuda() for k, v in targets.items()}]

        # Zero gradients for every batch!
        optimizer.zero_grad()

        print(targets)

        # Make predictions for this batch
        outputs = model(images, targets)
        print(outputs)
        exit()

        outputs = outputs.cpu()

        # Compute the loss and its gradients
        loss = loss_fn(outputs, target)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data
        running_loss += loss.item()

        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.0
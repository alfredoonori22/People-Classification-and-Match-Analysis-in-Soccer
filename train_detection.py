def train_one_epoch_detection(model, optimizer, training_loader, args):
    running_loss = 0.0

    # Make sure gradient tracking is on, and do a pass over the data
    model.train()

    for i, (images, targets) in enumerate(training_loader):

        # Every data instance is an image + target pair
        images = list(image.cuda() for image in images)
        targets = [{k: v.squeeze().cuda() for k, v in targets.items()}]

        # Make predictions for this batch
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Zero gradients for every batch!
        optimizer.zero_grad()

        losses.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data
        running_loss += losses.item()

        if i % 100 == 99:
            last_loss = running_loss / 10  # loss per batch
            print(f'  batch {i+1} loss: {last_loss}')
            running_loss = 0.0

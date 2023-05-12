import torch.nn


def train_one_epoch_detection(model, optimizer, training_loader, device, args):
    epochs = args.epochs
    batch_size = args.batch_size
    running_loss = 0.0
    last_loss = 0.0

    # Defining Loss
    loss_fn = torch.nn.CrossEntropyLoss()

    # Make sure gradient tracking is on, and do a pass over the data
    model.train()

    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs = inputs.cuda()

        # Zero gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)
        inputs = inputs.cpu()
        outputs = outputs.cpu()

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data
        running_loss += loss.item()

        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.0

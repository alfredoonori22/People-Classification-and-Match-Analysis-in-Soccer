import sys

import torch.utils.data

import transform as T
from datasets import SNDetection, Football_People
from models import our_CNN, create_fasterrcnn
from train_detection import train_one_epoch_cnn, evaluate_cnn, test_cnn
from utils import create_dataloader


def detection_cnn(args, folder):
    # Create model
    model = our_CNN()
    model.cuda()

    nn = "cnn"

    # Choosing split
    if args.train:
        # Initialization and Configuration wandb
        """
            wandb.init(project="SoccerNet",
                   name="Detection CNN",
                   config={
                       'multiclass': args.multiclass,
                       'dropout': args.dropout,
                       'architecture': 'our CNN'
                   })
        """
        print('Train phase for our CNN')

        # Data Loading Code
        print('Loading Data for Detection Training')
        dataset_train = Football_People(args, split='train', transform=T.get_transform("train", nn))
        dataset_valid = Football_People(args, split='valid', transform=T.get_transform("valid", nn))

        training_loader = create_dataloader(dataset_train, args.batch_size)
        validation_loader = create_dataloader(dataset_valid, args.batch_size)

        # Optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        best_score = 0
        counter = 0

        # Resuming
        if args.resume:
            print("Resuming")
            checkpoint = torch.load(f"{folder}/checkpoint_detection")
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            args.start_epoch = checkpoint['epoch'] + 1

            best_model = torch.load(f"{folder}/best_model")
            best_score = best_model['score']

        print("Start training")

        for epoch in range(args.start_epoch, args.epochs):
            print(f'EPOCH: {epoch + 1}')

            loss = train_one_epoch_cnn(model, optimizer, training_loader, epoch, folder)
            # Define our custom x axis metric
            """
            wandb.define_metric("epoch")
            wandb.define_metric("training_loss", step_metric='epoch', goal='minimize')
            # Update wandb
            wandb.log({'training_loss': loss, 'epoch': epoch})
            """
            # Save the model at the end of the epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, f"{folder}/checkpoint_detection")

            print("Start validation")
            # Validation at the end of each epoch
            score = evaluate_cnn(model, validation_loader)

            score = round(float(score) * 100, 2)

            # Update wandb
            """wandb.define_metric("validation_mAP", step_metric='epoch', goal='maximize')
            wandb.log({'validation_mAP': score, 'epoch': epoch})
            """
            print(f'valid F1 Score: {score}')

            # If score is better than the saved one, update it and save the model
            if score > best_score:
                print("---New best---")
                best_score = score
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'score': score,
                }, f"{folder}/best_model")

                counter = 0
            else:
                counter += 1

            if counter == args.patience:
                print('Early stropping')
                break

        # wandb.finish()

    if args.test:
        print('Test phase for our CNN')

        # Data Loading Code
        print('Loading Data for Detection test')
        # We want just the label to be in the multiclass form, not the model
        args.multiclass = True
        dataset_test = SNDetection(args, split='test', transform=T.get_transform("test", "fasterrcnn"))

        print("Creating data loader")
        test_loader = create_dataloader(dataset_test, 1)

        print("Retrieving the Faster-RCNN model")
        fasterrcnn = create_fasterrcnn(dropout=False, backbone=True, num_classes=3)
        best_model = torch.load(f"models/backbone/best_model")
        fasterrcnn.load_state_dict(best_model['model_state_dict'])

        # Retrieving the model
        print("Retrieving the CNN model")
        best_model = torch.load(f"{folder}/best_model")
        model.load_state_dict(best_model['model_state_dict'])

        print('Testing the model')
        # score = evaluate_cnn(model, test_loader)
        score = test_cnn(fasterrcnn, model, test_loader)
        score = round(float(score) * 100, 2)

        print(f'Test mAP: {score}')

    if not (args.train or args.test):
        sys.exit("Error: invalid split given")

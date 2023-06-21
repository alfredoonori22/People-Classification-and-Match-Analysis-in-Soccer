import sys
import torch.utils.data
import transform as T
from datasets import SNDetection
from models import create_fasterrcnn
from train_detection import train_one_epoch_fasterrcnn, evaluate_fasterrcnn, test_fasterrcnn
from utils import create_dataloader


def detection_fasterrcnn(args, folder):
    # Create model
    num_classes = 5 if args.multiclass else 3
    model = create_fasterrcnn(dropout=args.dropout, train_backbone=args.train_backbone, num_classes=num_classes)

    # Variable containg the model
    nn = "fasterrcnn"

    # Choosing split
    if args.train:
        # Initialization and Configuration wandb
        """wandb.init(project="SoccerNet",
                   name="Detection Faster-RCNN",
                   config={
                       'multiclass': args.multiclass,
                       'dropout': args.dropout,
                       'architecture': 'fasterrcnn_resnet50_fpn'
                   })"""
        print('Train phase for Detection task')

        # Data Loading Code
        print('Loading Data for Detection Training')
        dataset_train = SNDetection(args, split='train', transform=T.get_transform("train", nn))
        dataset_valid = SNDetection(args, split='valid', transform=T.get_transform("valid", nn))

        training_loader = create_dataloader(dataset_train, args.batch_size)
        validation_loader = create_dataloader(dataset_valid, 1)

        # Optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        best_score = 0
        counter = 0

        # Resuming from checkpoint
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

            loss = train_one_epoch_fasterrcnn(model, optimizer, training_loader, epoch, folder)

            # Define our custom x axis metric
            """wandb.define_metric("epoch")
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
            score = evaluate_fasterrcnn(model, validation_loader)
            score = round(float(score) * 100, 2)

            # Update wandb
            """wandb.define_metric("validation_mAP", step_metric='epoch', goal='maximize')
            wandb.log({'validation_mAP': score, 'epoch': epoch})
            """
            print(f'valid mAP: {score}')

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
                # Increment counter if model didn't improve his score
                counter += 1

            if counter == args.patience:
                print('Early stropping')
                break

        # wandb.finish()

    if args.test:
        print('Test phase for Detection task')

        print(f"Retrieving the model from {folder}")
        best_model = torch.load(f"{folder}/best_model")
        model.load_state_dict(best_model['model_state_dict'])

        print("Testing the model")
        test_fasterrcnn(model)

    if not (args.train or args.test):
        sys.exit("Error: invalid split given")

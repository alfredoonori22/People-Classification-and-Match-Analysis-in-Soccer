import math
import os
import sys

import torch.utils.data
import wandb
from torch import nn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn, faster_rcnn

import transform as T
from argument_parser import get_args
from dataset import SNDetection, collate_fn
from train_detection import train_one_epoch_detection, evaluate

os.environ["WANDB_SILENT"] = "true"


if __name__ == '__main__':
    args = get_args()
    print('These are the parameters from the command line: ')
    print(args)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        print('No cuda device')

    # Creating model
    print('Creating Model')
    kwargs = {"tau_l": args.tl, "tau_h": args.th, "trainable_backbone_layers": args.trainable_backbone_layers}
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT, **kwargs)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, 9)  # 9 is the numer of dataset classes (8) + 1 (background)

    # Adding dropout to the 2 fully connected layer
    model.roi_heads.box_head.fc6 = nn.Sequential(
        model.roi_heads.box_head.fc6,
        nn.Dropout(p=0.25))

    model.roi_heads.box_head.fc7 = nn.Sequential(
        model.roi_heads.box_head.fc7,
        nn.Dropout(p=0.25))

    # Freezing backbone and FPN
    model.backbone.requires_grad_(False)

    model.cuda()

    # Choosing split
    if args.train:
        # Initialization and Configuration wandb
        wandb.init(project="SoccerNet",
                   name="SoccerNet Detection",
                   config={
                       'dataset': 'SoccerNet',
                       'epochs': args.epochs,
                       'batch_size': args.batch_size,
                       'architecture': 'fasterrcnn_resnet50_fpn'
                   })
        print('Train phase for Detection task')

        # Data Loading Code
        print('Loading Data for Detection Training')
        dataset_train = SNDetection(args, split='train', transform=T.get_transform("train"))
        dataset_valid = SNDetection(args, split='valid', transform=T.get_transform("valid"))

        # Create data loaders for our datasets
        training_batch_sampler = torch.utils.data.BatchSampler(
            torch.utils.data.RandomSampler(dataset_train), batch_size=args.batch_size, drop_last=True)
        valid_batch_sampler = torch.utils.data.BatchSampler(
            torch.utils.data.RandomSampler(dataset_valid), batch_size=args.batch_size, drop_last=True)

        training_loader = torch.utils.data.DataLoader(dataset_train, batch_sampler=training_batch_sampler,
                                                      collate_fn=collate_fn)
        validation_loader = torch.utils.data.DataLoader(dataset_valid, batch_sampler=valid_batch_sampler,
                                                        collate_fn=collate_fn)

        # Optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        # Resuming
        if args.resume:
            print("Resuming")
            checkpoint = torch.load("model/checkpoint_detection")
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            args.start_epoch = checkpoint['epoch'] + 1
            loss = checkpoint['loss']

        print("Start training")

        # best_model = torch.load("model/best_model")
        # best_score = best_model['score']
        best_score = 0
        counter = 0

        for epoch in range(args.start_epoch, args.epochs):
            print(f'EPOCH: {epoch + 1}')

            loss = train_one_epoch_detection(model, optimizer, training_loader, epoch)

            # Define our custom x axis metric
            wandb.define_metric("epoch")
            wandb.define_metric("training_loss", step_metric='epoch', goal='minimize')
            # Update wandb
            wandb.log({'training_loss': loss, 'epoch': epoch})

            # Save the model at the end of the epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, "model/checkpoint_detection")

            print("Start validation")
            # Validation at the end of each epoch
            score = evaluate(model, validation_loader)
            score = float(score) * 100
            # score = round(float(score)*100, 2)

            # Update wandb
            wandb.define_metric("validation_mAP", step_metric='epoch', goal='maximize')
            wandb.log({'validation_mAP': score, 'epoch': epoch})

            print(f'valid mAP: {score}')

            # If score is better than the saved one, update it and save the model
            if math.isfinite(score) and score > best_score:
                print("New best")
                best_score = score
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'score': score,
                }, "model/best_model")

                counter = 0
            else:
                counter += 1

            if counter == args.patience:
                print('Early stropping')
                break

        wandb.finish()

    if args.test:
        print('Test phase for Detection task')

        # Data Loading Code
        print('Loading Data for Detection Test')
        dataset_test = SNDetection(args, split='test', transform=T.get_transform("test"))

        print("Creating data loader")
        test_batch_sampler = torch.utils.data.BatchSampler(
            torch.utils.data.SequentialSampler(dataset_test), batch_size=args.batch_size, drop_last=True)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_sampler=test_batch_sampler,
                                                  collate_fn=collate_fn)

        # Retrieving the model
        print("Retrieving the model")
        best_model = torch.load("model/best_model")
        model.load_state_dict(best_model['model_state_dict'])

        print('Testing the model')
        score = evaluate(model, test_loader)
        score = round(float(score)*100, 2)

        print(f'Test mAP: {score}')

    if not (args.train or args.test):
        sys.exit("Error: invalid split given")

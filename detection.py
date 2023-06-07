import os
import sys
import torch.utils.data
import wandb
from torch import nn
from argument_parser import get_args
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn, faster_rcnn
import transform as T
from dataset import SNDetection, collate_fn, create_dataloader, Football_People
from train_detection import train_one_epoch_detection, evaluate

os.environ["WANDB_SILENT"] = "true"


def CreateModel():
    print('Creating Model')
    model = fasterrcnn_resnet50_fpn(weights_backbone=ResNet50_Weights.IMAGENET1K_V1)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    # Multi-class model
    model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, 5)  # 5 is the numer of dataset classes (4) + 1 (background)
    # Two-class model
    """model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, 3)  # 3 is the numer of dataset classes (2) + 1 (background)"""

    if args.dropout:
        # Adding dropout to the 2 fully connected layer
        model.roi_heads.box_head.fc6 = nn.Sequential(
            model.roi_heads.box_head.fc6,
            nn.Dropout(p=0.15))

        model.roi_heads.box_head.fc7 = nn.Sequential(
            model.roi_heads.box_head.fc7,
            nn.Dropout(p=0.15))

    # Freezing backbone
    model.backbone.requires_grad_(False)

    model.cuda()

    return model


if __name__ == '__main__':
    args = get_args()
    print('These are the parameters from the command line: ')
    print(args)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        print('No cuda device')

    model = CreateModel()

    # Two-class version
    """folder = "model"""
    # Multi-class version
    folder = "model_multi"

    if args.dropout:
        """folder = "model_dropout"""
        folder = "model_multi_dropout"

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
        dataset_train = SNDetection(args, split='train', transform=T.get_transform("train", "detection"))
        dataset_valid = SNDetection(args, split='valid', transform=T.get_transform("valid", "detection"))

        training_loader = create_dataloader(dataset_train, args.batch_size)
        validation_loader = create_dataloader(dataset_valid, args.batch_size)

        # Optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        # Resuming
        if args.resume:
            print("Resuming")
            checkpoint = torch.load(f"{folder}/checkpoint_detection")
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            args.start_epoch = checkpoint['epoch'] + 1
            loss = checkpoint['loss']

        print("Start training")

        best_model = torch.load(f"{folder}/best_model")
        best_score = best_model['score']
        # best_score = 0
        counter = 0

        for epoch in range(args.start_epoch, args.epochs):
            print(f'EPOCH: {epoch + 1}')

            loss = train_one_epoch_detection(model, optimizer, training_loader, epoch, folder)

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
            }, f"{folder}/checkpoint_detection")

            print("Start validation")
            # Validation at the end of each epoch
            score = evaluate(model, validation_loader)
            score = round(float(score)*100, 2)

            # Update wandb
            wandb.define_metric("validation_mAP", step_metric='epoch', goal='maximize')
            wandb.log({'validation_mAP': score, 'epoch': epoch})

            print(f'valid mAP: {score}')

            # If score is better than the saved one, update it and save the model
            if score > best_score:
                print("New best")
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

        wandb.finish()

    if args.test:
        print('Test phase for Detection task')

        # Data Loading Code
        print('Loading Data for Detection Test')
        dataset_test = SNDetection(args, split='test', transform=T.get_transform("test", "detection"))

        print("Creating data loader")
        test_batch_sampler = torch.utils.data.BatchSampler(
            torch.utils.data.SequentialSampler(dataset_test), batch_size=args.batch_size, drop_last=True)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_sampler=test_batch_sampler,
                                                  collate_fn=collate_fn)

        # Retrieving the model
        print("Retrieving the model")
        best_model = torch.load(f"{folder}/best_model")
        model.load_state_dict(best_model['model_state_dict'])

        print('Testing the model')
        score = evaluate(model, test_loader)
        score = round(float(score)*100, 2)

        print(f'Test mAP: {score}')

    if not (args.train or args.test):
        sys.exit("Error: invalid split given")

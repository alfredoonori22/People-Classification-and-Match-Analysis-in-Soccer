import math
import time
from datetime import datetime

import torch.utils.data
from torch import nn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn, faster_rcnn

import transform as T
from argument_parser import get_args
from dataset import SNDetection, collate_fn
from train_detection import train_one_epoch_detection, validation

if __name__ == '__main__':
    args = get_args()
    print('These are the parameters from the command line: ')
    print(args)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        print('No cuda device')

    # Choosing task
    if args.task == 'detection':
        print('The chosen task is Detection')

        if args.split == 'train':
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

            print('Creating Model')
            kwargs = {"tau_l": args.tl, "tau_h": args.th, "trainable_backbone_layers": args.trainable_backbone_layers}
            model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT, **kwargs)

            # get number of input features for the classifier
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            # replace the pre-trained head with a new one
            model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, dataset_train.num_classes() + 1)

            # Adding dropout to the 2 fully connected layer
            model.roi_heads.box_head.fc6 = nn.Sequential(
                model.roi_heads.box_head.fc6,
                nn.Dropout(p=0.25))

            model.roi_heads.box_head.fc7 = nn.Sequential(
                model.roi_heads.box_head.fc7,
                nn.Dropout(p=0.25))

            model.cuda()

            # Optimizer
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

            # Resuming
            if args.resume:
                print("Resuming")
                checkpoint = torch.load("model/checkpoint_detection.pt")
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                args.start_epoch = checkpoint['epoch'] + 1
                loss = checkpoint['loss']

            print("Start training")
            start_time = time.time()
            timestamp = datetime.now().strftime('%d-%m_%H:%M')
            best_score = 0.0
            counter = 0

            for epoch in range(args.start_epoch, args.epochs):
                print(f'EPOCH: {epoch + 1}')

                loss = train_one_epoch_detection(model, optimizer, training_loader, epoch)

                # Save the model at the end of the epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, "model/checkpoint_detection.pt")

                print("Entering validation")
                # Validation at the end of each epoch
                score = validation(model, validation_loader)
                score = round(float(score*100), 2)

                print(f'LOSS valid {score}')

                # If score is better than the saved one, update it and save the model
                if math.isfinite(score) and score > best_score:
                    print("New best")
                    best_score = score
                    model_path = f'model/{timestamp}__{epoch}__{score}'
                    torch.save(model.state_dict(), model_path)

                    counter = 0
                else:
                    counter += 1

                if counter == args.patience:
                    print('Early stropping')
                    break

        else:
            print('Test phase for Detection task')
            # chiamata a funzione test da definire

    elif args.task == 'calibration':
        print('The chosen task is Camera Calibration')
    else:
        print('The chosen task is Image Retrieval')

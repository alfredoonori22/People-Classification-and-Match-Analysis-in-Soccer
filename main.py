from torchvision.models import ResNet50_Weights
from argument_parser import get_args
from train_detection import train_one_epoch_detection
from dataset import SNDetection
import os
import errno
import time
import torch.utils.data
from torchvision.models.detection import fasterrcnn_resnet50_fpn

if __name__ == '__main__':
    args = get_args()
    print('These are the parameters from the command line: ')
    print(args)

    if args.output_dir:
        # Try to create the output directory, if it already exists the code does nothing
        # else it raises an error
        try:
            os.makedirs(args.output_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

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

            dataset_train = SNDetection(args.data_path, split='train', tiny=args.tiny)
            dataset_valid = SNDetection(args.data_path, split='valid', tiny=args.tiny)

            # Create data loaders for our datasets
            training_batch_sampler = torch.utils.data.BatchSampler(
                torch.utils.data.RandomSampler(dataset_train), batch_size=args.batch_size, drop_last=True)
            valid_batch_sampler = torch.utils.data.BatchSampler(
                torch.utils.data.RandomSampler(dataset_valid), batch_size=args.batch_size, drop_last=True)

            training_loader = torch.utils.data.DataLoader(dataset_train, batch_sampler=training_batch_sampler)
            validation_loader = torch.utils.data.DataLoader(dataset_valid, batch_sampler=valid_batch_sampler)

            print('Creating Model')
            kwargs = {"tau_l": args.tl, "tau_h": args.th}
            model = fasterrcnn_resnet50_fpn(num_classes=dataset_train.num_classes()+1, weights_backbone=ResNet50_Weights.DEFAULT, **kwargs)
            model.cuda()
            print('Model Created')

            # Optimizer
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

            print("Start training")
            start_time = time.time()
            for epoch in range(args.epochs):
                print(f'EPOCH: {epoch + 1}')

                train_one_epoch_detection(model, optimizer, training_loader, args)

        else:
            print('Test phase for Detection task')
            # chiamata a funzione test da definire

    elif args.task == 'calibration':
        print('The chosen task is Camera Calibration')
    else:
        print('The chosen task is Image Retrieval')
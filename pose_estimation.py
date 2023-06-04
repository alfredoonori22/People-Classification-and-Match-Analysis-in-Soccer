import os
import sys
import torch.utils.data
import wandb
from argument_parser import get_args
from dataset import collate_fn, MPIIDataset

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
    # TODO: chiamata a funzione per la creazione del modello, model.py

    # Choosing split
    if args.train:
        # Initialization and Configuration wandb
        wandb.init(project="SoccerNet",
                   name="SoccerNet Pose Estimation",
                   config={
                       'dataset': 'MPII',
                       'epochs': args.epochs,
                       'batch_size': args.batch_size,
                       'architecture': 'mymodel'
                   })
        print('Train phase for Pose Estimation task')

        # Data Loading Code
        print('Loading Data for Pose Estimation Training')
        # TODO: costruire una classe per caricamento dataset
        dataset_train = MPIIDataset(split='train')
        dataset_valid = MPIIDataset(split='valid')

        # Create data loaders for our datasets
        training_batch_sampler = torch.utils.data.BatchSampler(
            torch.utils.data.RandomSampler('''dataset_train'''), batch_size=args.batch_size, drop_last=True)
        valid_batch_sampler = torch.utils.data.BatchSampler(
            torch.utils.data.RandomSampler('''dataset_valid'''), batch_size=args.batch_size, drop_last=True)

        training_loader = torch.utils.data.DataLoader('''dataset_train''', batch_sampler=training_batch_sampler,
                                                  collate_fn=collate_fn)
        validation_loader = torch.utils.data.DdataLoader('''dataset_valid''', batch_sampler=valid_batch_sampler,
                                                    collate_fn=collate_fn)

        # Optimizer
        params = [p for p in '''model.parameters()''' if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=args.lr)

        # Resuming
        if args.resume:
            print("Resuming")
            checkpoint = torch.load("model/checkpoint_p_estimation")
            '''model'''.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            args.start_epoch = checkpoint['epoch'] + 1
            loss = checkpoint['loss']

        print("Start training")

        # best_model = torch.load("model/best_model_pose")
        # best_score = best_model['score']
        counter = 0

        for epoch in range(args.start_epoch, args.epochs):
            print(f'EPOCH: {epoch + 1}')

            # TODO: scrivere la train_one_epoch per il Pose Estimation
            # loss = train_one_epoch_pose_estimation(model, optimizer, training_loader, epoch)

            # Define our custom x axis metric
            wandb.define_metric("epoch")
            wandb.define_metric("training_loss", step_metric='epoch', goal='minimize')
            # Update wandb
            wandb.log({'training_loss': loss, 'epoch': epoch})

            # Save the model at the end of the epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': '''model'''.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, "model/checkpoint_pose_estimation")

            print("Start validation")

            # Validation at the end of each epoch
            # TODO: Definire la funzione evaluate per il Pose Estimation
            # score = evaluate(model, validation_loader)
            # score = round(float(score) * 100, 2)

            # Update wandb
            # TODO: Stabilire la metrica da utilizzare per pose estimation
            # wandb.define_metric("validation_mAP", step_metric='epoch', goal='maximize')
            # wandb.log({'validation_mAP': score, 'epoch': epoch})

            # print(f'valid mAP: {score}')

            '''
            # If score is better than the saved one, update it and save the model
            if score > best_score:
                print("New best")
                best_score = score
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'score': score,
                }, "model/best_model_pose")

                counter = 0
            else:
                counter += 1

            if counter == args.patience:
                print('Early stropping')
                break
            '''

        wandb.finish()

    if args.test:
        print('Test phase for Pose Estimation task')

        # Data Loading Code
        print('Loading Data for Pose Estimation Test')
        # dataset_test = <function>(args, split='test', transform=<transforms_to_apply>)

        print("Creating data loader")
        test_batch_sampler = torch.utils.data.BatchSampler(
            torch.utils.data.SequentialSampler('''dataset_test'''), batch_size=args.batch_size, drop_last=True)
        test_loader = torch.utils.data.DataLoader('''dataset_test''', batch_sampler=test_batch_sampler,
                                                  collate_fn=collate_fn)

        # Retrieving the model
        print("Retrieving the model")
        best_model = torch.load("model/best_model_pose")
        '''model'''.load_state_dict(best_model['model_state_dict'])

        print('Testing the model')
        # TODO: Definire la funzione evaluate per il Pose Estimation
        # score = evaluate(model, test_loader)
        # score = round(float(score)*100, 2)

        # print(f'Test mAP: {score}')

    if not (args.train or args.test):
        sys.exit("Error: invalid split given")


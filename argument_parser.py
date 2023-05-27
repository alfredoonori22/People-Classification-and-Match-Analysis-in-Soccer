import argparse     # parsing elements from command line


def get_args():
    # Creating an object that will contain the information given from the command line
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--data-path', default='/mnt/beegfs/work/cvcs_2022_group20/SoccerNet-v3', help='Dataset path')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--test-only', action='store_true', help='Only test the model')
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('-b', '--batch-size', default=4, type=int)
    parser.add_argument('--tiny', required=False, type=int, default=None, help='Select a subset of x games')
    parser.add_argument('--size', type=str, default="(1920,1080)", help='Target dimension for image pre-processing')

    # Model
    parser.add_argument('--trainable-backbone-layers', type=int, default=0, help='number of trainable layers of backbone')
    parser.add_argument('--tl', default=0.9, type=float, help='Value for tau_low (default: 0.9')
    parser.add_argument('--th', default=1., type=float, help='Value for tau_high (default: 1.)')

    # Training
    parser.add_argument('--resume', type=str, default=False, help='Resume from checkpoint')
    parser.add_argument('--start_epoch', type=int, default=0, help='Start epoch')
    parser.add_argument('--patience', type=int, default=5, help='Number of epochs before early stopping')

    # Optimizer
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('-wd', '--weight-decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')

    args = parser.parse_args()
    return args

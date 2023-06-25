import argparse     # parsing elements from command line


def get_args():
    # Creating an object that will contain the information given from the command line
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--data-path', default='/mnt/beegfs/work/cvcs_2022_group20', help='Dataset path')
    parser.add_argument('--tiny', required=False, type=int, default=None, help='Select a subset of x games')
    parser.add_argument('--size', type=str, default="(1920,1080)", help='Target dimension for image pre-processing')

    # Training
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('-b', '--batch-size', default=4, type=int)
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--start-epoch', type=int, default=0, help='Start epoch')
    parser.add_argument('--patience', type=int, default=20, help='Number of epochs before early stopping')

    # Test
    parser.add_argument('--test', action='store_true', help='Test the model')
    parser.add_argument('-v', '--video-path', default='test.mp4', help='Test video path')

    # Model
    parser.add_argument('--model', type=str, default='fasterrcnn', help='fasterrcnn or cnn')
    parser.add_argument('--multiclass', action='store_true', help='If true differentiate between people')
    parser.add_argument('--dropout', action='store_true', help='Use dropout in fc layers')
    parser.add_argument('--train-backbone', action='store_true', help='Train backbone')

    # Optimizer
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('-wd', '--weight-decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')

    # Analysis
    parser.add_argument('--deep', action='store_true', help='Deep version of Geometry')

    args = parser.parse_args()
    return args

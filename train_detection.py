from dataset import get_dataset


def train_one_epoch_detection(args):
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = 0.001
    running_loss = 0.0
    last_loss = 0.0

    # Data Loading Code
    print('Loading Data for Detection Training')
    get_dataset(args.data_path, args.split)


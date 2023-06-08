import os
import sys

import torch.utils.data

from argument_parser import get_args
from detection_fasterrcnn import detection_fasterrcnn
from detection_ourCNN import detection_cnn

os.environ["WANDB_SILENT"] = "true"


if __name__ == '__main__':
    args = get_args()
    print('These are the parameters from the command line: ')
    print(args)

    if not torch.cuda.is_available():
        sys.exit('No cuda device')

    # Choosing the correct folder depending on the model
    if args.multiclass:
        if args.dropout:
            folder = "models/model_multi_dropout"
        else:
            folder = "models/model_multi"
    else:
        if args.dropout:
            folder = "models/model_dropout"
        else:
            folder = "models/model"

    if args.model == "cnn":
        folder = "model/cnn"

    if args.model == "fasterrcnn":
        detection_fasterrcnn(args, folder)
    elif args.model == "cnn":
        detection_cnn(args, folder)
    else:
        sys.exit("Error: invalid model given")

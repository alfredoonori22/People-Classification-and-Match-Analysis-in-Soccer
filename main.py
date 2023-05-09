from argument_parser import get_args
import os
import errno
import torch


if __name__ == '__main__':
    args = get_args()
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
        print('No cuda device detected')

    # Choosing task
    if args.task == 'detection':
        print('The chosen task is Detection')

        if args.split == 'train':
            print('Train phase for Detection task')
            # chiamata a funzione train da definire
        else:
            print('Test phase for Detection task')
            # chiamata a funzione test da definire

    elif args.task == 'calibration':
        print('The chosen task is Camera Calibration')
    else:
        print('The chosen task is Image Retrieval')
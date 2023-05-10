import argparse     # parsing elements from command line


def get_args():
    # Creating an object that will contain the information given from the command line
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--data-path', default='/mnt/beegfs/work/cvcs_2022_group20 ', help='dataset')
    parser.add_argument('--split', type=str, default='train', help='train or test')
    parser.add_argument('--output-dir', type=str, default='', help='directory where to save results, empty if no saving')
    parser.add_argument('--task', type=str, help='detection, calibration or retrieval')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch-size', default=4, type=int)

    args = parser.parse_args()
    return args

import torch.utils.data
import json
import os


def get_dataset(root, image_set):
    # Searching for the annotation file
    ann_file = '{}_annotations.json'.format(image_set)
    ann_file = os.path.join(root, ann_file)

    dataset = SNDetection(ann_file)

    if 'train' in image_set:
        num_classes = dataset.num_classes()
        return dataset, num_classes
    else:
        return dataset


# Building the dataset
class SNDetection(torch.utils.data.Dataset):
    print('starting the SNDetection creation')

    def __init__(self, path):
        self.path = path                            # Saving the path parameter
        self.data = json.load(open(path, 'r'))      # Loading the data from a JSON file specified by the path
        self.keys = list(self.data.keys())          # Creating a list of keys (attributes) from the data           
        self.targets = list()                       # Holding the targets associated with the data in the current object
        self.labels = list()                        # Holding the labels associated with the data in the current object

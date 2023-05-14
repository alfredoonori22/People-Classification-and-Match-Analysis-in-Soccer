import torch.utils.data
import json
import os
from tqdm import tqdm
from SoccerNet.utils import getListGames
from PIL import Image
import torch.utils.data
from torchvision.transforms import functional

CLASS_DICT = {'Ball': 1,
              'Player team left': 2,
              'Player team right': 3,
              'Goalkeeper team left': 4,
              'Goalkeeper team right': 5,
              'Main referee': 6,
              'Side referee': 7,
              'Staff members': 8
             }

"""
class CheckBoxes:
    def __call__(self, image, target):
        w, h = image.size

        boxes = target['boxes']
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        target['boxes'] = boxes

        return image, target

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
"""

# Building the dataset
class SNDetection(torch.utils.data.Dataset):
    def __init__(self, path,split, tiny):
        # Per il momento non fa nulla, poi servirà per le operazioni di transform
        #t = [CheckBoxes()]
        #transform = Compose(t)
        #self.transforms = transform

        self.path = path

        # Get the list of the selected subset of games
        self.list_games = getListGames(split, task="frames")
        if tiny is not None:
            self.list_games = self.list_games[:tiny]

        self.data = list()
        for game in tqdm(self.list_games):
            self.data.append(json.load(open(os.path.join(self.path, game, "Labels-v3.json"))))

        self.targets = list()
        self.labels = list()

        # Variable for the total length of the dataset
        self.tot_len = 0

        # Variable that stores the full name of each image (ex, 'path/0.png')
        self.full_keys = []

        for i,game in enumerate(self.list_games):
            # Loop through the game

            self.keys = list(self.data[i]['actions'].keys())     # List of images of each game
            self.tot_len = self.tot_len + len(self.keys)

            for k in self.keys:
                # Loop through the images
                self.full_keys.append(f'{self.path}/{game}/Frames-v3/{k}')

                boxes = list()
                labels = list()
                image_id = k.split('.')[0]

                for b in self.data[i]['actions'][k]['bboxes']:
                    # Loop through the bboxes of each image
                    # Verify if the bbox's label is in the dictionary
                    if b['class'] not in CLASS_DICT:
                        continue

                    # Creating a list of image's labels
                    self.labels.append(CLASS_DICT[b['class']])
                    labels.append(CLASS_DICT[b['class']])

                    # Creating a list of dictionaries with the points of the bboxes
                    boxes.append([b['points']['x1'],b['points']['x2'],b['points']['y1'],b['points']['y2']])

                self.targets.append({'boxes': torch.tensor(boxes),
                                    'labels': torch.tensor(labels),
                                    'image_id': torch.tensor(int(image_id))})


    def __len__(self, ):
        return self.tot_len

    def __getitem__(self, idx):

        image = Image.open(self.full_keys[idx]).convert('RGB')
        targets = self.targets[idx]
        #image, targets = self.transforms(image, targets)

        return functional.to_tensor(image), targets
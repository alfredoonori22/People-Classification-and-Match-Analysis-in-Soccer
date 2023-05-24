import ast
import json
import os

import torch.utils.data
from PIL import Image
from SoccerNet.utils import getListGames
from tqdm import tqdm

import transform as T

CLASS_DICT = {'Ball': 1,
              'Player team left': 2,
              'Player team right': 3,
              'Goalkeeper team left': 4,
              'Goalkeeper team right': 5,
              'Main referee': 6,
              'Side referee': 7,
              'Staff members': 8
              }


def collate_fn(batch):
    return tuple(zip(*batch))


# Building the dataset
class SNDetection(torch.utils.data.Dataset):
    def __init__(self, args, split, transform=None):

        self.path = args.data_path

        self.size = ast.literal_eval(args.size)

        # t will be the list containg all the pre-process operation, Resize and CheckBoxes are always present here
        t = [T.ResizeImg(), T.CheckBoxes()]
        # if there are other operation they are appended
        if transform is not None:
            t.extend(transform)
        # Compose make the list a callable
        self.transforms = T.Compose(t)

        # Get the list of the selected subset of games
        self.list_games = getListGames(split, task="frames")
        if args.tiny is not None:
            # Dimiuiamo dimensione validation per debug
            if split == "valid":
                args.tiny = args.tiny // 5
            self.list_games = self.list_games[:args.tiny]

        self.data = list(json.load(open(os.path.join(self.path, f"Labels-{split}.json"))))
        self.targets = list()
        self.labels = list()

        # Variable for the total length of the dataset
        self.tot_len = 0

        # Variable that stores the full name of each image (ex, 'path/0.png')
        self.full_keys = list()

        for i, game in enumerate(tqdm(self.list_games)):
            # Loop through the game

            self.keys = list(self.data[i]['actions'].keys())  # List of images of each game (actions)
            self.keys.extend(list(self.data[i]['replays'].keys()))   # and replays images
            self.tot_len = self.tot_len + len(self.keys)

            for k in self.keys:
                # Loop through the images
                self.full_keys.append(f'{self.path}/{game}/Frames-v3/{k}')

                boxes = list()
                labels = list()
                areas = list()
                iscrowds = list()

                # List containg the bboxes, its value changes depending on the key (k, action/replay)
                iterate = list()
                # If it's a replay (0_0.png) get the bboxes from 'replays' key in the json, else from 'actions'
                if "_" in k:
                    iterate.extend(self.data[i]['replays'][k]['bboxes'])
                    # Image ID saved as a float to store information about its replay number
                    # E.g. 0_0 is the first replay of action 0, and it's saved as a float --> 0.1
                    image_id = f'{k.split(".")[0].split("_")[0]}.{int(k.split(".")[0].split("_")[1]) + 1}'

                    # Get the original Image size
                    x = self.data[i]['replays'][k]['imageMetadata']['width']
                    y = self.data[i]['replays'][k]['imageMetadata']['height']
                else:
                    # Image Id of action 0 is simply 0.0
                    iterate.extend(self.data[i]['actions'][k]['bboxes'])
                    image_id = k.split('.')[0]

                    # Get the original Image size
                    x = self.data[i]['actions'][k]['imageMetadata']['width']
                    y = self.data[i]['actions'][k]['imageMetadata']['height']

                # Get the target size to know the scale factors
                targetSize = self.size
                x_scale = targetSize[0] / x
                y_scale = targetSize[1] / y

                for b in iterate:
                    # Loop through the bboxes of each image

                    # Verify if the bbox's label is in the dictionary
                    if b['class'] not in CLASS_DICT:
                        continue

                    # Descard degenerate bboxes (we assure that xmin < xmax and the same for y)
                    if (b['points']['x2'] <= b['points']['x1']) or (b['points']['y2'] <= b['points']['y1']):
                        continue
                    else:
                        # Creating a list with the points of the bboxes
                        boxes.append([b['points']['x1'], b['points']['y1'], b['points']['x2'], b['points']['y2']])

                    # Creating a list of image's labels
                    self.labels.append(CLASS_DICT[b['class']])
                    labels.append(CLASS_DICT[b['class']])

                    # Calculate bbox area
                    a = (b['points']['x2'] - b['points']['x1']) * (b['points']['y2'] - b['points']['y1'])
                    areas.append(a)
                    iscrowds.append(0)

                # Scale the bboxes
                boxes = torch.tensor(boxes)
                boxes[:, 0::2] = boxes[:, 0::2] * x_scale
                boxes[:, 1::2] = boxes[:, 1::2] * y_scale

                tmp_dict = {'boxes': boxes,
                            'labels': torch.tensor(labels),
                            'image_id': torch.tensor(float(image_id)),
                            'area': torch.tensor(areas),
                            'iscrowd': torch.tensor(iscrowds)
                            }

                self.targets.append(tmp_dict)

        self.labels = [i for i in set(self.labels) if i > 0]

    def num_classes(self, ):
        return len(self.labels)

    def __len__(self, ):
        return self.tot_len

    def __getitem__(self, idx):

        image = Image.open(self.full_keys[idx]).convert('RGB')
        target = self.targets[idx]

        image, target = self.transforms(image, target, self.size)

        return image, target

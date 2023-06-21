import ast
import json
import os
import torch.utils.data
from PIL import Image
from SoccerNet.utils import getListGames
import transform as T
from utils import CLASS_DICT, MULTI_CLASS_DICT, PEOPLE_DICT


# Building the dataset
class SNDetection(torch.utils.data.Dataset):
    def __init__(self, args, split, transform=None):

        self.path = os.path.join(args.data_path, 'SoccerNet-v3')

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
            # Reduce validation set's size
            if split == "valid":
                args.tiny = args.tiny // 5
            self.list_games = self.list_games[:args.tiny]

        self.data = list(json.load(open(os.path.join(self.path, f"Labels-{split}.json"))))
        self.targets = list()
        self.labels = list()

        # Variable that stores the full name of each image (ex, 'path/0.png')
        self.full_keys = list()

        for i, game in enumerate(self.list_games):
            # Loop through the game

            self.keys = list(self.data[i]['actions'].keys())  # List of images of each game (actions)

            for k in self.keys:
                # Loop through the images
                self.full_keys.append(f'{self.path}/{game}/Frames-v3/{k}')

                boxes = list()
                labels = list()
                areas = list()
                iscrowds = list()

                # Image Id of action 0 is simply 0.0
                image_id = k.split('.')[0]

                # Get the original Image size
                x = self.data[i]['actions'][k]['imageMetadata']['width']
                y = self.data[i]['actions'][k]['imageMetadata']['height']

                # Get the target size to know the scale factors
                targetSize = self.size
                x_scale = targetSize[0] / x
                y_scale = targetSize[1] / y

                for b in self.data[i]['actions'][k]['bboxes']:
                    # Loop through the bboxes of each image

                    # Merge label for two type of Player, Goalkeeper and Referee
                    if args.multiclass:
                        # Multi-class version
                        if b['class'].startswith("Player"):
                            b['class'] = "Player"
                        elif b['class'].startswith("Goalkeeper"):
                            b['class'] = "Goalkeeper"
                        elif b['class'].endswith("referee"):
                            b['class'] = "Referee"

                        if b['class'] not in MULTI_CLASS_DICT:
                            continue

                        # Creating a list of image's labels
                        self.labels.append(MULTI_CLASS_DICT[b['class']])
                        labels.append(MULTI_CLASS_DICT[b['class']])
                    else:
                        # Two-class version
                        if b['class'].endswith("left") | b['class'].endswith("right") | b['class'].endswith("referee"):
                            b['class'] = "Person"

                        if b['class'] not in CLASS_DICT:
                            continue

                        # Creating a list of image's labels
                        self.labels.append(CLASS_DICT[b['class']])
                        labels.append(CLASS_DICT[b['class']])

                    # Descard degenerate bboxes (we assure that xmin < xmax and the same for y)
                    if (b['points']['x2'] <= b['points']['x1']) or (b['points']['y2'] <= b['points']['y1']):
                        continue
                    else:
                        # Creating a list with the points of the bboxes
                        boxes.append([b['points']['x1'], b['points']['y1'], b['points']['x2'], b['points']['y2']])

                    # Calculate bbox area
                    a = (b['points']['x2'] - b['points']['x1']) * (b['points']['y2'] - b['points']['y1'])
                    areas.append(a)
                    iscrowds.append(0)

                # Scale the bboxes
                boxes = torch.tensor(boxes)
                boxes[:, 0::2] = boxes[:, 0::2] * x_scale
                boxes[:, 1::2] = boxes[:, 1::2] * y_scale

                # Append target dictionary for this image
                self.targets.append({'boxes': boxes,
                                     'labels': torch.tensor(labels),
                                     'image_id': torch.tensor(float(image_id)),
                                     'area': torch.tensor(areas),
                                     'iscrowd': torch.tensor(iscrowds)})

        self.labels = [i for i in set(self.labels) if i > 0]

    def num_classes(self, ):
        return len(self.labels)

    def __len__(self, ):
        return len(self.targets)

    def __getitem__(self, idx):

        image = Image.open(self.full_keys[idx]).convert('RGB')
        target = self.targets[idx]

        # Pre-processing operations
        image, target = self.transforms(image, target, self.size)

        return image, target


# Building Dataset for Football People to recognize people classes
class Football_People(torch.utils.data.Dataset):
    def __init__(self, args, split, transform=None):

        self.path = os.path.join(args.data_path, 'Football_People')
        # Same size for all the images
        self.size = (40, 80)

        # t will be the list containg all the pre-process operation
        t = [T.ResizeImg()]
        # if there are other operation they are appended
        if transform is not None:
            t.extend(transform)
        # Compose make the list a callable
        self.transforms = T.Compose(t)

        # Read annotation file
        self.data = list(json.load(open(os.path.join(self.path, f"Labels-{split}.json"))))
        self.targets = list()
        self.full_keys = list()

        # Building the dataset
        for ann in self.data:
            image_name = ann['image_id']
            self.full_keys.append(os.path.join(f'{self.path}/{split}-images/{image_name}'))

            self.targets.append(torch.tensor(PEOPLE_DICT[ann['label']]))

    def __len__(self, ):
        return len(self.targets)

    def __getitem__(self, idx):

        image = Image.open(self.full_keys[idx]).convert('RGB')
        target = self.targets[idx]

        # Pre-processing operations
        image, target = self.transforms(image, target, self.size)

        return image, target

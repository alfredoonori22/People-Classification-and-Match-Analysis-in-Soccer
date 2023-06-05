import ast
import json
import os
import numpy as np
import torch.utils.data
from PIL import Image
from SoccerNet.utils import getListGames

import transform as T

CLASS_DICT = {'Ball': 1,
              'Person': 2}


def collate_fn(batch):
    return tuple(zip(*batch))


def create_dataloader(dataset, batch_size):
    batch_sampler = torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(dataset), batch_size=batch_size,
                                                  drop_last=True)
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn)

    return loader


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
            # Diminuiamo dimensione validation per debug
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

        for i, game in enumerate(self.list_games):
            # Loop through the game

            self.keys = list(self.data[i]['actions'].keys())  # List of images of each game (actions)
            self.tot_len = self.tot_len + len(self.keys)

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

                    if b['class'].endswith("left") | b['class'].endswith("right") | b['class'].endswith("referee"):
                        b['class'] = "Person"

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


class MPIIDataset(torch.utils.data.Dataset):
    def __init__(self, args, split, transform=None):

        self.path = os.path.join(args.data_path, 'MPII')
        self.num_joints = 16
        self.flip_pairs = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]
        self.parent_ids = [1, 2, 6, 6, 3, 4, 6, 6, 7, 8, 11, 12, 7, 7, 13, 14]
        self.split = split
        self.transform = transform

        # Reading annotation file
        file_name = os.path.join(f'{self.path}/mpii_annotations/mpii_{self.split}.json')
        self.data = list(json.load(open(file_name)))
        self.targets = list()
        self.full_keys = list()

        # Building the dataset
        for i, ann in enumerate(self.data):
            image_name = ann['image']
            self.full_keys.append(os.path.join(f'{self.path}/images/{image_name}'))

            c = np.array(ann['center'], dtype=float)
            s = np.array([ann['scale'], ann['scale']], dtype=float)

            # Adjust center/scale slightly to avoid cropping limbs
            if c[0] != -1:
                c[1] = c[1] + 15 * s[1]
                s = s * 1.25

            # MPII uses matlab format, index is based 1,
            # we should first convert to 0-based index
            c = c - 1

            joints_3d = np.zeros((self.num_joints, 3), dtype=float)
            joints_3d_vis = np.zeros((self.num_joints, 3), dtype=float)

            if self.split != 'test':
                joints = np.array(ann['joints'])
                joints[:, 0:2] = joints[:, 0:2] - 1
                joints_vis = np.array(ann['joints_vis'])
                assert len(joints) == self.num_joints, \
                    'joint num diff: {} vs {}'.format(len(joints),
                                                      self.num_joints)

                joints_3d[:, 0:2] = joints[:, 0:2]
                joints_3d_vis[:, 0] = joints_vis[:]
                joints_3d_vis[:, 1] = joints_vis[:]

            self.targets.append({
                'image': self.full_keys[i],
                'center': c,
                'scale': s,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis
            })

    def __len__(self, ):
        return len(self.targets)

    def __getitem__(self, idx):

        image = Image.open(self.full_keys[idx]).convert('RGB')
        target = self.targets[idx]

        # image, target = self.transforms(image, target, self.size)

        return image, target

import torch.utils.data
import json
import os
import copy
import torchvision.transforms
import zipfile
from tqdm import tqdm
from SoccerNet.utils import getListGames
from torchvision import transforms
from torchvision import io
from PIL import Image
from SoccerNet.Evaluation.utils import FRAME_CLASS_DICTIONARY
import torch.utils.data


# Building the dataset
class SNDetection(torch.utils.data.Dataset):
    print('starting the SNDetection creation')

    def __init__(self, path, split="all", resolution=(1920, 1080), preload_images=False, tiny=None,
                 zipped_images=False):

        # Path for the SoccerNet-v3 Dataset containing images and labels
        self.path = path

        # Get the list of the selected subset of games
        self.list_games = getListGames(split, task='frames')
        if tiny is not None:
            self.list_games = self.list_games[:tiny]

        # Resolution of the images to load (width, height)
        self.resolution = resolution
        self.resize = torchvision.transforms.Resize((resolution[1],resolution[0]), antialias=True)
        self.preload_images = preload_images
        self.zipped_images = zipped_images

        # Variable to store the metadata
        print('Reading the annotation files')
        self.metadata = list()
        for game in tqdm(self.list_games):
            self.metadata.append(json.load(open(os.path.join(self.path, game, "Labels-v3.json"))))

        # Variable to store the preloaded images and annotations
        # Each element in the list is a list of images and annotations linked to an action
        self.data = list()

        for annotations in tqdm(self.metadata):

            # Retrieve each action in the game
            for action_name in annotations['GameMetadata']['list_actions']:

                # Concatenate the replays of each action with itself
                img_list = [action_name] + annotations['actions'][action_name]['linked_replays']
                self.data.append(list())
                IDs_list = list()

                zipfilepath = os.path.join(self.path,annotations["GameMetadata"]["UrlLocal"], 'Frames-v3.zip')
                if self.zipped_images:
                    zippedFrames = zipfile.ZipFile(zipfilepath, 'r')

                # For each image extract the images and annotations
                for i, img in enumerate(img_list):

                    # Variable to save the annotation
                    data_tmp = dict()
                    data_tmp['image'] = None

                    # Only the first frame is an action, the rest are replays
                    img_type = 'actions'
                    if i > 0:
                        img_type = 'replays'

                    filepath = os.path.join(self.path,annotations["GameMetadata"]["UrlLocal"], "Frames-v3", img)
                    if self.preload_images:
                        with torch.no_grad():
                            if self.zipped_images:
                                imginfo = zippedFrames.open(img)
                                data_tmp['image'] = self.resize(transforms.ToTensor()(Image.open(imginfo))*255)
                            else:
                                data_tmp['image'] = self.resize(torchvision.io.read_image(filepath))

                    data_tmp['zipfilepath'] = zipfilepath
                    data_tmp['imagefilepath'] = img
                    data_tmp['filepath'] = filepath

                    data_tmp['bboxes'], ID_tmp = self.format_bboxes(annotations[img_type][img]['bboxes'],
                                                                    annotations[img_type][img]['imageMetadata'])

                    IDs_list.append(ID_tmp)

                    self.data[-1].append(data_tmp)

    def __len__(self, ):
        return len(self.list_games)

    def format_bboxes(self, bboxes, image_metadata):

        # Bounding boxes in x_top, y_top, width, height, cls_idx, num_idx
        data = list()
        IDs = list()

        for i, bbox in enumerate(bboxes):

            if bbox['class'] is not None:

                tmp_data = torch.zeros((4 + 1 + 1,), dtype=torch.float) - 1
                tmp_data[0] = bbox['points']['x1'] / image_metadata['width']
                tmp_data[1] = bbox['points']['y1'] / image_metadata['height']
                tmp_data[2] = abs(bbox['points']['x2'] - bbox['points']['x1']) / image_metadata['width']
                tmp_data[3] = abs(bbox['points']['y2'] - bbox['points']['y1']) / image_metadata['height']
                tmp_data[4] = float(FRAME_CLASS_DICTIONARY[bbox['class']])

                if bbox['ID'] is not None:
                    if bbox['ID'].isnumeric():
                        tmp_data[5] = float(bbox['ID'])

                IDs.append([bbox['ID'], FRAME_CLASS_DICTIONARY[bbox['class']]])
                data.append(tmp_data)

        data = torch.stack(data)
        return data, IDs

    def __getitem__(self, index):

        if not self.preload_images:
            data = copy.deepcopy(self.data[index])
            with torch.no_grad():
                for i, d in enumerate(data):
                    if self.zipped_images:
                        imginfo = zipfile.ZipFile(d['zipfilepath'], 'r').open(d['imagefilepath'])
                        img = transforms.ToTensor()(Image.open(imginfo)) * 255
                        data[i]['image'] = self.resize(img)

                    else:
                        data[i]['image'] = self.resize(torchvision.io.read_image(d['filepath']))
                return data

        return self.data[index]

import cv2
import numpy as np
import torchvision
import torch
from torch.utils.data import BatchSampler, DataLoader
from torchmetrics.detection import MeanAveragePrecision
from torchvision.transforms import ToPILImage

# Model with only two classes
CLASS_DICT = {'Ball': 1,
              'Person': 2}

# Multi-class model
MULTI_CLASS_DICT = {'Ball': 1,
                    'Player': 2,
                    'Goalkeeper': 3,
                    'Referee': 4}

PEOPLE_DICT = {'Player': 0,
               'Goalkeeper': 1,
               'Referee': 2}

BBOX_COLORS = {1: [0, 0, 255],    # Ball --> Red
               2: [255, 0, 0],    # Players --> Blue
               3: [0, 85, 10],    # Goalkeeper --> Green
               4: [0, 0, 0]}      # Referee --> Black


def collate_fn(batch):
    return tuple(zip(*batch))


def create_dataloader(dataset, batch_size):
    batch_sampler = BatchSampler(torch.utils.data.RandomSampler(dataset), batch_size=batch_size, drop_last=True)
    loader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn)

    return loader


def draw_bbox(image, output, classes=3):
    image = image.cpu()
    image = ToPILImage()(image)
    image = np.array(image)

    keys = list(CLASS_DICT.keys()) if classes == 3 else list(MULTI_CLASS_DICT.keys())

    for i, (x1, y1, x2, y2) in enumerate(output['boxes']):
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), BBOX_COLORS[int(output['labels'][i])], 2)
        if classes == 3:
            index = (list(CLASS_DICT.values()).index(int(output['labels'][i])))
        else:
            index = (list(MULTI_CLASS_DICT.values()).index(int(output['labels'][i])))

        cv2.putText(image, keys[index], (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    BBOX_COLORS[int(output['labels'][i])], 2)

    return image


def apply_nms(orig_prediction, iou_thresh=0.3, thresh=0.3):
    # Indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]

    # Keep only the output's bboxes with a score higher than the threshold
    if torch.any(final_prediction['scores'] > thresh):
        output = {'boxes': torch.stack([box for box, label, score in zip(final_prediction['boxes'], final_prediction['labels'], final_prediction['scores'])
                                        if score > thresh]),
                  'labels': torch.tensor([label for label, score in zip(final_prediction['labels'], final_prediction['scores'])
                                          if score > thresh]),
                  'scores': torch.tensor([score for score, label in zip(final_prediction['scores'], final_prediction['labels'])
                                          if score > thresh])}
    else:
        output = {'boxes': torch.FloatTensor([]),
                  'labels': torch.FloatTensor([]),
                  'scores': torch.FloatTensor([])}

    return output


def xyxy2xywh(bbox):
    x1 = bbox[0]
    y1 = bbox[1]
    w = bbox[2] - x1
    h = bbox[3] - y1

    return [x1, y1, w, h]


def ourMetric(outputs, targets):
    metric = MeanAveragePrecision(class_metrics=True, iou_thresholds=[0.5, 0.75])
    metric.update(outputs, targets)
    res = metric.compute()

    # Number of ball and people bboxes found
    num_balls = len(np.where(outputs[0]['labels'] == 1)[0])
    num_people = len(np.where(outputs[0]['labels'] != 1)[0])

    if res['map_per_class'].numel() == 1:
        return res['map_per_class']

    # If there's the ball, the first element in res['map_per_class'] is ball's mAP,
    # then we take only the classes' mAP that are positive (because when a class is not found in the image its mAP is -1)
    if num_balls:
        map_ball = res['map_per_class'][0] * num_balls
        res_people = [x for x in res['map_per_class'][1:].numpy() if x > 0]
        map_people = np.sum(res_people) / len(res_people) * num_people if len(res_people) else 0

        # weighted mAP between ball mAP and people mAP,
        # the class' weight is given by by the number of element in the image of that category
        mAP = (map_ball + map_people) / (num_balls + num_people)
    else:
        # If there are no ball in the image all the element in res['map_per_class] refer to people classes
        res_people = [x for x in res['map_per_class'].numpy() if x > 0]
        mAP = np.sum(res_people) / len(res_people) if len(res_people) else 0

    return mAP

import cv2
import numpy as np
import torch
import torchvision
from torch.utils.data import BatchSampler, DataLoader
from torchvision.transforms import ToPILImage

"""
from torchvision.transforms import ToPILImage
image1 = box_img.cpu()
image1 = ToPILImage()(image1)
image1.save("test-images/image1.png")"""

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


def collate_fn(batch):
    return tuple(zip(*batch))


def create_dataloader(dataset, batch_size):
    batch_sampler = BatchSampler(torch.utils.data.RandomSampler(dataset), batch_size=batch_size, drop_last=True)
    loader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn)

    return loader


def draw_bbox(image, target, output):
    image = image.cpu()
    image = ToPILImage()(image)
    image = np.array(image)
    # keys = list(CLASS_DICT.keys())
    keys = list(MULTI_CLASS_DICT.keys())

    for i, (x1, y1, x2, y2) in enumerate(output['boxes']):
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        # index = (list(MULTI_CLASS_DICT.values()).index(int(output['labels'][i])))
        index = (list(MULTI_CLASS_DICT.values()).index(int(output['labels'][i])))
        cv2.putText(image, keys[index], (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    for (x1, y1, x2, y2) in target['boxes']:
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

    cv2.imwrite(f"test-images/bbox-{target['image_id']}.png", image)


def apply_nms(orig_prediction, iou_thresh=0.3):
    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]

    if torch.any(final_prediction['scores'] > 0.3):
        output = {'boxes': torch.stack([box for box, score in zip(final_prediction['boxes'], final_prediction['scores'])
                                        if score > 0.3]),
                  'labels': torch.tensor([label for label, score in zip(final_prediction['labels'], final_prediction['scores'])
                                          if score > 0.3]),
                  'scores': torch.tensor([score for score in final_prediction['scores']
                                          if score > 0.3])}
    else:
        output = {'boxes': torch.FloatTensor([]),
                  'labels': torch.FloatTensor([]),
                  'scores': torch.FloatTensor([])}

    return output

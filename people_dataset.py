import json
import os

import cv2
from SoccerNet.utils import getListGames
from tqdm import tqdm

if __name__ == '__main__':
    path = os.path.join('/mnt/beegfs/work/cvcs_2022_group20/SoccerNet-v3')
    # Get the list of the selected subset of games
    list_games = getListGames('valid', task="frames")
    list_games = list_games[:11]
    data = list(json.load(open(os.path.join(path, f"Labels-valid.json"))))
    labels = []

    height = 0
    width = 0
    count = 0

    for i, game in enumerate(tqdm(list_games)):
        # Loop through the game

        keys = list(data[i]['actions'].keys())  # List of images of each game (actions)

        for k in keys:
            # Loop through the images
            match = f'{game}/{k}'
            image = cv2.imread(f'/mnt/beegfs/work/cvcs_2022_group20/SoccerNet-v3/{game}/Frames-v3/{k}')

            for j, b in enumerate(data[i]['actions'][k]['bboxes']):
                # Loop through the bboxes of each image

                # Merge label for two type of Player, Goalkeeper and Referee
                if b['class'].startswith("Player"):
                    b['class'] = "Player"
                elif b['class'].startswith("Goalkeeper"):
                    b['class'] = "Goalkeeper"
                elif b['class'].endswith("referee"):
                    b['class'] = "Referee"
                else:
                    continue

                # Descard degenerate bboxes (we assure that xmin < xmax and the same for y)
                if (b['points']['x2'] <= b['points']['x1']) or (b['points']['y2'] <= b['points']['y1']):
                    continue
                if int(b['points']['y1']) < 0 | int(b['points']['y2']) < 0 \
                        | int(b['points']['x1']) < 0 | int(b['points']['x2']) < 0:
                    continue

                height = height + (int(b['points']['y2']) - int(b['points']['y1']))
                width = width + (int(b['points']['x2']) - int(b['points']['x1']))
                count = count + 1

                box_img = image[int(b['points']['y1']):int(b['points']['y2']), int(b['points']['x1']):int(b['points']['x2'])]
                name = f"{i}_{k.split('.')[0]}_{j}.png"
                cv2.imwrite(f"/mnt/beegfs/work/cvcs_2022_group20/Football_People/valid-images/{name}", box_img)

                tmp_dict = {
                    "image_id": name,
                    "set": "valid",
                    "label": b["class"]}

                labels.append(tmp_dict)

    with open("/mnt/beegfs/work/cvcs_2022_group20/Football_People/Labels-valid.json", "w") as file:
        json.dump(labels, file, indent=2)
    file.close()

    print(f"L'altezza media è {height/count}")
    print(f"L'ampiezza media è {width/count}")

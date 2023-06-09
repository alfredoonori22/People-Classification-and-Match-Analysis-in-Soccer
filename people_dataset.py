import json
import os

import cv2
from SoccerNet.utils import getListGames
from tqdm import tqdm

if __name__ == '__main__':
    path = os.path.join('/mnt/beegfs/work/cvcs_2022_group20/SoccerNet-v3')
    # Get the list of the selected subset of games
    split = "train"
    list_games = getListGames(split, task="frames")
    data = list(json.load(open(os.path.join(path, f"Labels-{split}.json"))))
    labels = []

    count_p = 0
    count_g = 0
    count_r = 0

    for i, game in enumerate(tqdm(list_games)):
        # Loop through the game

        keys = list(data[i]['actions'].keys())  # List of images of each game (actions)

        for k in keys:
            # Loop through the images
            image = cv2.imread(f'/mnt/beegfs/work/cvcs_2022_group20/SoccerNet-v3/{game}/Frames-v3/{k}')

            for j, b in enumerate(data[i]['actions'][k]['bboxes']):
                # Loop through the bboxes of each image
                # Descard degenerate bboxes (we assure that xmin < xmax and the same for y)
                if (b['points']['x2'] <= b['points']['x1']) or (b['points']['y2'] <= b['points']['y1']):
                    continue
                if int(b['points']['y1']) < 0 | int(b['points']['y2']) < 0 \
                        | int(b['points']['x1']) < 0 | int(b['points']['x2']) < 0:
                    continue

                # Merge label for two type of Player, Goalkeeper and Referee
                if b['class'].startswith("Player") and count_p < 6450:
                    count_p = count_p + 1
                    b['class'] = "Player"
                elif b['class'].startswith("Goalkeeper") and count_g < 6450:
                    b['class'] = "Goalkeeper"
                    count_g = count_g + 1
                elif b['class'].endswith("referee") and count_r < 6450:
                    b['class'] = "Referee"
                    count_r = count_r + 1
                else:
                    continue

                box_img = image[int(b['points']['y1']):int(b['points']['y2']), int(b['points']['x1']):int(b['points']['x2'])]
                if not box_img.any():
                    continue

                name = f"{i}_{k.split('.')[0]}_{j}.png"
                cv2.imwrite(f"/mnt/beegfs/work/cvcs_2022_group20/Football_People/{split}-images/{name}", box_img)

                labels.append({"image_id": name,
                               "set": f"{split}",
                               "label": b["class"]})

    with open(f"/mnt/beegfs/work/cvcs_2022_group20/Football_People/Labels-{split}.json", "w") as file:
        json.dump(labels, file, indent=2)
    file.close()

    print(f"I giocatori in {split} sono {count_p}")
    print(f"I portieri in {split} sono {count_g}")
    print(f"Gli arbitri in {split} sono {count_r}")

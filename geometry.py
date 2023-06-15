import cv2
import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from torchvision.transforms import functional
from argument_parser import get_args
from models import create_fasterrcnn
from utils import apply_nms, xyxy2xywh

BALL_DIAMETER = 23


def PeopleDetection(zoomed_image):
    # HOG Descriptor for People Detection
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Pre-processing operations
    gray = cv2.cvtColor(zoomed_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # Detect people's bboxes
    bboxes, scores = hog.detectMultiScale(gray, winStride=(1, 1), padding=(5, 5), scale=1.05)
    bboxes = [bbox.tolist() for bbox, score in zip(bboxes, scores) if score > 0.6]

    return bboxes


def NearestPlayer(bboxes, ball_center, image):

    # Saving in a list all bboxes centers
    centers = []

    for i, _ in enumerate(bboxes):
        x1, y1 = bboxes[i][0], bboxes[i][1]

        center_x = int(x1 + (bboxes[i][2]) / 2)
        center_y = int(y1 + (bboxes[i][3]) / 2)

        centers.append((center_x, center_y))

    # Calculate distance from the ball's center and each player's bbox center
    distances = cdist(np.array([ball_center]), np.array(centers), 'euclidean')[0]

    # Find player index
    nearest = distances.argmin()
    # Distance in pixel for the nearest player
    distance_nearest = float(distances[nearest])

    # Drawing nearest player bboxes
    cv2.rectangle(image, (int(bboxes[nearest][0]), int(bboxes[nearest][1])),
                  (int(bboxes[nearest][0] + bboxes[nearest][2]), int(bboxes[nearest][1] + bboxes[nearest][3])),
                  (255, 0, 0), 2)

    return nearest, distance_nearest


def px2cm(distance, width):
    measure = 'cm'

    # cm/pixel ratio
    pixel_ratio = BALL_DIAMETER / width
    dist = round(pixel_ratio * distance, 2)

    # Distance in meters
    if dist > 100:
        dist = round(dist / 100, 2)
        measure = 'm'

    return dist, measure


def ShirtColor(image, bbox):
    # Find bbox coordinates
    x1, y1 = int(bbox[0]), int(bbox[1])
    x2, y2 = int(x1 + bbox[2]), int(y1 + bbox[3])

    # Crop the image on the player
    player_img = image[y1:y2, x1:x2]

    # New dimension of the image
    height, width, _ = player_img.shape

    # Crop the image on the shirt
    shirt_img = player_img[int(height / 8):int(1 * height / 2), int(width / 4):int(3 * width / 4)]

    # New dimension of the image
    height, width, dim = shirt_img.shape

    # Convert the image in HSV
    hsv_image = cv2.cvtColor(shirt_img, cv2.COLOR_BGR2HSV)

    # Range for gray and green in HSV
    lower_gray = np.array([0, 0, 0], dtype=np.uint8)
    upper_gray = np.array([179, 50, 200], dtype=np.uint8)
    lower_green = np.array([40, 40, 40], dtype=np.uint8)
    upper_green = np.array([70, 255, 255], dtype=np.uint8)

    # Create the two masks
    mask_gray = cv2.inRange(hsv_image, lower_gray, upper_gray)
    mask_green = cv2.inRange(hsv_image, lower_green, upper_green)

    # Invert the masks
    inverse_mask_gray = cv2.bitwise_not(mask_gray)
    inverse_mask_green = cv2.bitwise_not(mask_green)

    # Apply masks to the image
    result_gray = cv2.bitwise_and(shirt_img, shirt_img, mask=inverse_mask_gray)
    result = cv2.bitwise_and(result_gray, result_gray, mask=inverse_mask_green)

    # Vectorize the result image
    img_vec = np.reshape(result, [height * width, dim])

    # Using KMeans algorithm to obtain pixel's color clusters
    kmeans = KMeans(n_clusters=7, n_init=10)
    kmeans.fit(img_vec)
    unique_l, counts_l = np.unique(kmeans.labels_, return_counts=True)

    # Sort the clusters' vector from biggest to smallest
    sort_ix = np.argsort(counts_l)
    sort_ix = sort_ix[::-1]

    # Pick the second cluster's color in BGR format (the first will always be background color)
    bgr_color = kmeans.cluster_centers_[sort_ix][1].round(0).astype(int)

    return bgr_color


def interpolate_frames(last_player):
    bboxes = output['boxes'][np.where(output['labels'] == 2)]
    bboxes = [xyxy2xywh(bbox) for bbox in bboxes]
    NearestPlayer(bboxes, last_player, cv_image)


if __name__ == '__main__':
    args = get_args()

    # Retrieving best model
    model = create_fasterrcnn(dropout=False, backbone=True, num_classes=3)
    best_model = torch.load('/mnt/beegfs/homes/aonori/SoccerNet/models/backbone/best_model')
    model.load_state_dict(best_model['model_state_dict'])
    model.eval()

    # Load input video
    video = cv2.VideoCapture('/mnt/beegfs/work/cvcs_2022_group20/test.mp4')
    # video.set(cv2.CAP_PROP_FPS, 30)
    success, cv_image = video.read()
    # Create output video
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15, (cv_image.shape[1], cv_image.shape[0]))

    last_player = (0, 0)
    last_ball = (0, 0)
    first = True

    while True:
        # Read frame from video
        success, cv_image = video.read()
        if not success:
            break

        # Image pre-processing
        image = functional.to_tensor(cv_image)
        image = image.cuda()

        # Finding boxes in image
        with torch.inference_mode():
            output = model([image])

        output = {k: v.cpu() for k, v in output[0].items()}
        # Non Max Suppression to discard intersected superflous bboxes
        output = apply_nms(output, iou_thresh=0.2, thresh=0.75)

        # Predicted ball boxes
        ball_boxes = output['boxes'][np.where(output['labels'] == 1)]
        ball_scores = output['scores'][np.where(output['labels'] == 1)]

        # Skip image without ball
        if len(ball_boxes) == 0:
            if not args.deep:
                cv2.putText(cv_image, "Ball not found", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 3)
                print("Ball not found")
            else:
                if not first:
                    # Track last player
                    interpolate_frames(last_player)
                    print("Ball not found, tracking last player found")

            out.write(cv_image)
            continue

        # Select the ball box with the highest score
        idx = np.argmax(ball_scores)
        ball_box = ball_boxes[idx]

        # Ball coordinates
        x1, y1, x2, y2 = ball_box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Ball's center coordinates in original image
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        if not first:
            # Distance between new ball and the last ball found
            distance = cdist(np.array([last_ball]), np.array([(center_x, center_y)]), 'euclidean')[0]
            if distance > 50:
                if args.deep:
                    interpolate_frames(last_player)
                out.write(cv_image)
                print('Ball is too distant from last position')
                continue

        last_ball = (center_x, center_y)
        first = False

        # Ball's bbox dimension
        width_ball = x2 - x1
        height_ball = y2 - y1

        if not args.deep:
            # Zoom factor
            zoom_factor = 10

            # Zooming the image on the ball
            new_width = int(width_ball * zoom_factor)
            new_height = int(height_ball * zoom_factor)

            # New dimension of the image
            x1_im = max(0, center_x - int(new_width / 2))
            y1_im = max(0, center_y - int(new_height / 2))
            x2_im = min(cv_image.shape[1], x1_im + new_width)
            y2_im = min(cv_image.shape[0], y1_im + new_height)
            zoomed_image = cv_image[y1_im:y2_im, x1_im:x2_im]

            # Ball's center coordinates in resized image
            ball_center = (new_width / 2, new_height / 2)

            # Detect people in zoomed image
            bboxes = PeopleDetection(zoomed_image)

            if len(bboxes) == 0:
                cv2.putText(cv_image, "No player nearby", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 3)
                out.write(cv_image)
                print("No player nearby")
                continue
        else:
            # Take the bboxes of players
            bboxes = output['boxes'][np.where(output['labels'] == 2)]
            bboxes = [xyxy2xywh(bbox) for bbox in bboxes]
            ball_center = (center_x, center_y)
            zoomed_image = cv_image

        # Find the nearest Player from the ball
        idx, distance_px = NearestPlayer(bboxes, ball_center, zoomed_image)

        # Update last player found
        last_player = (int(bboxes[idx][0]+bboxes[idx][2]/2), int(bboxes[idx][1]+bboxes[idx][3]/2))

        # Ball
        cv2.rectangle(cv_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

        # Convert pixel distance to cm distance
        distance_cm, measure = px2cm(distance_px, width_ball)

        # Find dominant shirt color
        color = ShirtColor(zoomed_image, bboxes[idx])

        cv2.putText(cv_image, f'The closest player is {distance_cm} {measure} from the ball\nHis shirt color is BGR: '
                              f'{color}', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 3)

        print(f'The closest player is {distance_cm} {measure} from the ball')
        print(f'His shirt color is BGR: {color}')
        out.write(cv_image)

    # Release resources
    video.release()
    out.release()
    cv2.destroyAllWindows()

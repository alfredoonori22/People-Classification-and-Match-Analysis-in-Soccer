import cv2
import numpy as np
from argument_parser import get_args
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

BALL_DIAMETER = 23

def PeopleDetection():
    # HOG Descriptor for People Detection
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Pre-processing operations
    gray = cv2.cvtColor(zoomed_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # Detect people's bboxes
    bboxes, _ = hog.detectMultiScale(gray, winStride=(1, 1), padding=(5, 5), scale=1.1)

    # Drawing bboxes
    for (x, y, w, h) in bboxes:
        cv2.rectangle(zoomed_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return bboxes


def NearestPlayer(bboxes, ball_center):

    # Saving in a list all bboxes centers
    centers = []

    for i, _ in enumerate(bboxes):
        x1, y1 = bboxes[i][0], bboxes[i][1]

        center_x = int(x1 + (bboxes[i][2]) / 2)
        center_y = int(y1 + (bboxes[i][3]) / 2)

        centers.append((center_x, center_y))

    # Calculate distance from the ball's center and each player's bbox center
    distance = cdist(np.array([ball_center]), np.array(centers), 'euclidean')

    # Find the index of the player
    nearest = distance.argmin()

    # Distance in pixel from the nearest player
    distance_nearest = float(distance[nearest])

    return nearest, distance_nearest


def px2cm(distance, width):
    measure = 'cm'

    # Ratio pixel - cm
    pixel_ratio = BALL_DIAMETER / width
    dist = round(pixel_ratio * distance, 2)

    # Distance in meters
    if dist > 100:
        dist = round(dist / 100, 2)
        measure = 'm'

    return dist, measure


def ShirtColor(image, bbox):
    # Find bbox coordinates
    x1, y1 = bbox[0], bbox[1]
    x2, y2 = x1 + bbox[2], y1 + bbox[3]

    # Crop the image on the player
    player_img = image[y1:y2, x1:x2]

    # New dimension of the image
    height, width, _ = player_img.shape

    # Crop the image on the shirt
    shirt_img = player_img[int(height / 8):int(3 * height / 8), int(width / 4):int(3 * width / 4)]

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


if __name__ == '__main__':
    args = get_args()
    print('These are the parameters from the command line: ')
    print(args)

    # TODO: invece di avere un solo frame, estraiamo i frame da un video test
    image = cv2.imread(
        f'{args.data_path}/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/Frames-v3/1.png')

    # Bounding box ball
    x1, x2, y1, y2 = int(677.33), int(698.85), int(502.3), int(524.51)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Dimension of ball's bb
    width_ball = x2 - x1
    height_ball = y2 - y1

    # Zoom factor
    zoom_factor = 10

    # Zooming the image
    new_width = int(width_ball * zoom_factor)
    new_height = int(height_ball * zoom_factor)

    # Ball's center coordinates in original image
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)

    # New dimension of the image
    x1 = max(0, center_x - int(new_width / 2))
    y1 = max(0, center_y - int(new_height / 2))
    x2 = min(image.shape[1], x1 + new_width)
    y2 = min(image.shape[0], y1 + new_height)
    zoomed_image = image[y1:y2, x1:x2]

    # Ball's center coordinates in resized image
    ball_center = (new_width / 2, new_height / 2)

    # HOG Descriptor for People Detection
    bboxes = PeopleDetection()

    # Nearest Player
    idx, distance_px = NearestPlayer(bboxes, ball_center)

    # Convert pixel distance to cm distance
    distance_cm, measure = px2cm(distance_px, width_ball)

    # Find dominant shirt color
    color = ShirtColor(zoomed_image, bboxes[idx])

    print(f'The nearest player is at {distance_cm} {measure} from the ball')
    print(f'His shirt color is BGR: {color}')

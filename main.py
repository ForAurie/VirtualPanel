import mediapipe as mp
import cv2, time
import numpy as np

############################
mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

MARGIN = 3  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
#   handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    # handedness = handedness_list[idx]

    # Draw the hand landmarks.
    mp_drawing.draw_landmarks(
      annotated_image,
      hand_landmarks,
      mp_hands.HAND_CONNECTIONS,
      mp_drawing_styles.get_default_hand_landmarks_style(),
      mp_drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    # cv2.putText(annotated_image, f"{handedness[0].category_name}",
    #             (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
    #             FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image
############################

cap = cv2.VideoCapture(0)

if not cap:
   print("Cannot open camera.")
   exit()

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

def calc_cos(landmark, a : int, b : int, c : int, d : int) -> float:
    a, b, c, d = landmark[a], landmark[b], landmark[c], landmark[d]
    a, b, c, d = np.array([a.x, a.y, a.z]), np.array([b.x, b.y, b.z]), np.array([c.x, c.y, c.z]), np.array([d.x, d.y, d.z])
    x, y, z = a - b, b - c, c - d
    x /= np.linalg.norm(x)
    y /= np.linalg.norm(y)
    z /= np.linalg.norm(z)
    return np.sqrt(max(np.dot(x, y) * np.dot(y, z), 0))

def is_grabbed(detections : HandLandmarkerResult) -> list[bool]:
    result = [(
       calc_cos(i, 12, 11, 10, 9) + 
       calc_cos(i, 16, 15, 14, 13) + 
       calc_cos(i, 20, 19, 18, 17)
       ) < 1.8 for i in detections.hand_world_landmarks]
    return result

global positions, scale, last_grab
positions = [np.array([0.5, 0.5]) for i in range(5)]
last_grab = dict()
scale = 0.02

def result_callback(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global positions, scale, last_grab
    grab = is_grabbed(result)
    annotated_image = draw_landmarks_on_image(output_image.numpy_view(), result)
    h, w = annotated_image.shape[:2]
    norm_l = 0.5 * (w + h) * scale
    new_grab = dict()
    for i in range(len(grab)):
        idx = result.handedness[i][0].index
        point = np.array([result.hand_landmarks[i][8].x, result.hand_landmarks[i][8].y])
        if last_grab.get(idx) is not None:
            if grab[i]:
                positions[last_grab.get(idx)] = point
                new_grab[idx] = last_grab.get(idx)
        elif grab[i]:
            for j in range(len(positions)):
                dis = np.abs(point - positions[j])
                if dis[0] * w < norm_l and dis[1] * h < norm_l:
                    positions[j] = point
                    new_grab[idx] = j
                    break
    last_grab = new_grab
    for pos in positions:
        x = round(pos[0] * w)
        y = round(pos[1] * h)
        x, y = max(x, 0), max(y, 0)
        x, y = min(x, w), min(y, h)
        cv2.circle(annotated_image, (x, y), max(2, round(norm_l)), (0, 255, 0), thickness=2)
    cv2.imshow("VirtualPanel", cv2.flip(annotated_image, 1))
    cv2.waitKey(1)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=result_callback,
    num_hands=2)
with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        landmarker.detect_async(mp_image, int(time.time() * 1000))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()
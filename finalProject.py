# Fully functional local version of the squat counter
# Runs on a virtual environment with Python 3.10 and ultralytics 8.0.201

''' 
- EXERCISE -
Modify the program so that it counts squats in three different modes. In the first mode, 
it should count them as it is done in the original. In the second mode, it should count 
them only and exclusively if the hands are above the hips throughout the entire squat. 
In the third mode, it will be the same as the second, but with the hands above the shoulders.

The counting mode should appear on the screen (mode 1, 2, or 3), and each time a squat is 
performed and it is not correct according to the mode, a message should appear on the screen 
that says "bad squatting."
'''
##################################################
# Libraries
##################################################

#!/usr/bin/env python 

import math
from collections import deque
import cv2
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO


##################################################
# Global variables and general set up
##################################################

# Body parts ordered as indicated in keypoints
idx2bparts = ["Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip", "Left Knee",
    "Right Knee", "Left Ankle", "Right Ankle"]

# Index of body parts
bparts2idx = {key: ix for ix, key in enumerate(idx2bparts)}

# State and squat count
STATE = 'UP'
COUNT = 0
state_stack = deque(maxlen=6)
CHECK = True  # Used for debugging
ONE_IMAGE = False
MODE = 1  # Default mode
PROGRESS = 0  # Squat progress
HANDS_POSITION_WARNING = False # Warning dsplayed when the hands are not in the right position for squat

# Load the Yolov8 model
model = YOLO('src/models/yolov8s-pose.pt')

# Open the video file
source = 0
# source = "src/inference/videos/MySquats.mp4"
video_mode2_path = "MySquats.mp4"
video_mode3_path = "MySquats.mp4"

##################################################
# Helper functions
##################################################

def add_annotations(frame):
    """
    Add state (up/down) and squats count (number) to the image.

    Args:
        frame (numpy array): Current frame captured

    Returns:
        frame with added text
    """
    # Display state and count on the image
    state_text = f"State: {STATE}"
    count_text = f"Count: {COUNT}"
    mode_text = f"Mode: {MODE}"
    progress_text = f"Progress: {PROGRESS:.0f}%"


    # Define the position and font settings for the text
    text_position1 = (10, 30)
    text_position2 = (10, 60)
    text_position3 = (10, 90)
    text_position4 = (10, 120)
    warning_position = (10, 200)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    green = (0, 255, 0)
    red = (0, 0, 255)
    blue = (255, 0, 0)
    font_color = green if STATE == 'UP' else red
    font_thickness = 2

    frame_with_text = frame.copy()
    cv2.putText(frame_with_text, state_text, text_position1, font,
                font_scale, font_color, font_thickness)
    cv2.putText(frame_with_text, count_text, text_position2, font,
                font_scale, green, font_thickness)
    cv2.putText(frame_with_text, mode_text, text_position3, font,
                font_scale, green, font_thickness)
    cv2.putText(frame_with_text, progress_text, text_position4, font,
                font_scale, blue, font_thickness)
    
    if HANDS_POSITION_WARNING:
        warning_text = "Hands not above the shoulders" if MODE == 3 else "Hands not above the hip"
        cv2.putText(frame_with_text, warning_text, warning_position, font,
                    font_scale, red, font_thickness)
    # Progress bar
    bar_x, bar_y, bar_w, bar_h = 10, 150, 200, 20
    progress_w = int(bar_w * PROGRESS / 100)
    cv2.rectangle(frame_with_text, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), blue, 2)
    cv2.rectangle(frame_with_text, (bar_x, bar_y), (bar_x + progress_w, bar_y + bar_h), blue, -1)

    return frame_with_text


def legs_angles(left, right, verbose=False):
    """
    It calculates the minimum angle that make up the vector hip-knee with
    the vector knee-ankle in each leg. The inputs are numpy arrays with
    shape 3x2 (3 points x 2 coordinates) and the output is a numpy array
    of shape [2,] with each angle in degrees.

    Args:
        left (numpy array): Coordinates of joints hip, knee and ankle of
            the left leg. The matrix has the following shape:
            [x hip  , y hip  ]
            [x knee , y knee ]
            [x ankle, y ankle]
        right (numpy array): Coordinates of joints hip, knee and ankle of
            the right leg. The matrix has the same shape as 'left'
        verbose (bool, optional): Print info. Defaults to False.

    Returns:
        A numpy array with shape [2,] with the angles of the two legs in
            degrees.
    """

    angles = []

    for v in [left, right]:
        # Define the coordinates of three points (x1, y1), (x2, y2), and (x3, y3)
        x1, y1 = v[0, 0], v[0, 1]
        x2, y2 = v[1, 0], v[1, 1]
        x3, y3 = v[2, 0], v[2, 1]

        # Calculate the vectors from p2 to p1 and from p2 to p3
        vector1 = (x1 - x2, y1 - y2)
        vector2 = (x3 - x2, y3 - y2)

        # Calculate the dot product of the vectors
        dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]

        # Calculate the magnitudes of the vectors
        magnitude1 = math.sqrt(vector1[0]**2 + vector1[1]**2)
        magnitude2 = math.sqrt(vector2[0]**2 + vector2[1]**2)

        # Calculate the cosine of the angle using the dot product
        cosine_theta = dot_product / (magnitude1 * magnitude2)

        # Calculate the angle in radians
        theta_radians = math.acos(max(-1, min(cosine_theta, 1)))

        # Convert the angle from radians to degrees
        theta_degrees = math.degrees(theta_radians)

        # Append the angles to the list
        angles.append(theta_degrees)

        if verbose:
            print((f"The angle in the knee (triangle knee-hip-ankle) is "
                   f"{theta_degrees:.2f} degrees."))

    return np.array(angles)


def get_legs_coords(kpts):
    """
    It gets the keypoints of the result object and extract those from
    hip, knee and ankle of left and right legs. The outputs are np arrays
    with the coordinates x, y and the confidence value

    Args:
        kpts (ultralytics keypoints): Keypoints object from the Result
            object in a pose estimation.

    Returns:
        left_leg_coords (numpy array): 3x3 numpy array with the coordinates
            (x, y, confidence) of the left hip, left knee and left ankle
            in the image
        left_leg_coords (numpy array): 3x3 numpy array with the coordinates
            (x, y, confidence) of the left hip, left knee and left ankle
            in the image
    """
    # Indices of left and right hip, knee and ankle
    left_leg = [11, 13, 15]
    right_leg = [12, 14, 16]

    # Left leg
    left_leg_coords = kpts.data[0, left_leg, :].cpu().numpy()

    # Right leg
    right_leg_coords = kpts.data[0, right_leg, :].cpu().numpy()

    return left_leg_coords, right_leg_coords

def get_hands_coords(kpts):
    """
    Get the coordinates of the left and right wrists.
    
    Args:
        kpts (ultralytics keypoints): Keypoints object from the Result
            object in a pose estimation.

    Returns:
        left_hand_coords (numpy array): Coordinates of the left wrist.
        right_hand_coords (numpy array): Coordinates of the right wrist.
    """
    # Indices of left and right wrists
    left_hand = [9]
    right_hand = [10]

    # Left hand
    left_hand_coords = kpts.data[0, left_hand, :].cpu().numpy()

    # Right hand
    right_hand_coords = kpts.data[0, right_hand, :].cpu().numpy()

    return left_hand_coords, right_hand_coords

def get_shoulders_coords(kpts):
    """
    Obtiene las coordenadas de los hombros izquierdo y derecho.

    Args:
        kpts (ultralytics keypoints): Objeto de puntos clave del resultado
            de una estimación de pose.

    Returns:
        left_shoulder_coords (numpy array): Coordenadas del hombro izquierdo.
        right_shoulder_coords (numpy array): Coordenadas del hombro derecho.
    """
    # Índices de los hombros izquierdo y derecho
    left_shoulder = [5]
    right_shoulder = [6]

    # Hombro izquierdo
    left_shoulder_coords = kpts.data[0, left_shoulder, :].cpu().numpy()

    # Hombro derecho
    right_shoulder_coords = kpts.data[0, right_shoulder, :].cpu().numpy()

    return left_shoulder_coords, right_shoulder_coords



def extract(result):
    """
    Explore the Results object of Ultralytics for pose estimation

    This is just a helper function in the sense it could help how to explore
    some fields in the Results objects. You won't really need this function
    to implement any functionality.

    Args:
        result (Ultralytics Results): Object extracted from a Results generator
            or a Results list.

    Returns:
        None. It prints out some info contained in the input object.
    """
    # Body parts ordered as indicated in keypoints
    idx2bparts = ["Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
        "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
        "Left Wrist", "Right Wrist", "Left Hip", "Right Hip", "Left Knee",
        "Right Knee", "Left Ankle", "Right Ankle"]

    # Index of body parts
    bparts2idx = {key: ix for ix, key in enumerate(idx2bparts)}

    # Process result generator
    output_str = ''
    for ix, r in enumerate(result):
        names = r.names

        # Boxes object for bbox outputs
        box = r.boxes
        output_str += "\n\nBOXES\n-----\n"
        output_str += f"Box {ix}\n"
        output_str += f"Name of object: {names[int(box.cls.item())]}\n"
        output_str += f"Normalized coordinates of the box (xyxy): {box.xyxyn}\n"
        output_str += f"Confidence of detection: {box.conf.item()}\n"

        kpts = r.keypoints  # Keypoints object for pose outputs
        output_str += "\n\nKEYPOINTS\n---------\n"
        output_str += "Coordinates normalized\n"
        for kp in kpts:
            output_str += f"Nose: {kp.xyn[0, bparts2idx['Nose']]}\n"
            output_str += f"Left Shoulder: {kp.xyn[0, bparts2idx['Left Shoulder']]}\n"
            output_str += f"Right Shoulder: {kp.xyn[0, bparts2idx['Right Shoulder']]}\n"
            output_str += f"Left Hip: {kp.xyn[0, bparts2idx['Left Hip']]}\n"
            output_str += f"Right Hip: {kp.xyn[0, bparts2idx['Right Hip']]}\n"
            output_str += f"Left Knee: {kp.xyn[0, bparts2idx['Left Knee']]}\n"
            output_str += f"Right Knee: {kp.xyn[0, bparts2idx['Right Knee']]}\n"
            output_str += f"Left Ankle: {kp.xyn[0, bparts2idx['Left Ankle']]}\n"
            output_str += f"Right Ankle: {kp.xyn[0, bparts2idx['Right Ankle']]}\n"

        print(output_str)

        # You could also explore masks and probs

        # Masks object for segmentation masks outputs
        masks = result.masks
        if masks is not None:
            output_str += "\n\nMASKS\n------\n"
            output_str += f"Number of masks: {len(masks.data)}\n"
            for i, mask in enumerate(masks.data):
                output_str += f"Mask {i} shape: {mask.shape}\n"

        # Probs object for classification outputs
        probs = r.probs
        if probs is not None:
            output_str += "\n\nPROBABILITIES\n-------------\n"
            for i, prob in enumerate(probs):
                output_str += f"Class {i} ({names[i]}): {prob:.4f}\n"


def evaluate_position(result, limit_conf=0.3, verbose=False, mode=1):
    """
    Evaluate position for mode 2: squats with hands above the hips

    Args:
        result (Ultralytics Results): Results object from Ultralytics. It
            contains all the data of the pose estimation.
        limit_conf (float, optional): It's the limiting confidence. Greater
            confidences in (all) points estimation will be considered,
            otherwise they will be discarded. Defaults to 0.3.
        verbose (bool, optional): Print info. Defaults to False.
    """

    # Global variables
    global COUNT
    global STATE
    global state_stack
    global PROGRESS
    global HANDS_POSITION_WARNING

    HANDS_POSITION_WARNING = False

    # Loop through Ultralytics Results
    for r in result:

        # Get bounding boxes
        box = r.boxes
        if r.names[int(box.cls.item())] != 'person':
            print("First box is not a person")
            break

        # Get keypoints
        kpts = r.keypoints  # Keypoints object for pose outputs

        # Get coordinates of the joints of the left and right legs
        left_coords, right_coords = get_legs_coords(kpts) #left_leg = [11, 13, 15] & right_leg = [12, 14, 16]
        left_hand_coords, right_hand_coords = get_hands_coords(kpts)
        left_shoulder_coords, right_shoulder_coords = get_shoulders_coords(kpts)


        # Geting hips coordinates:
        ''' How it works:
        Coordinates of the left hip (left_coords[0, :])
        Coordinates of the left knee (left_coords[1, :])
        Coordinates of the left ankle (left_coords[2, :])
        '''
        left_hip_y = left_coords[0, 1] # Accesing to coordinate y of left hip
        right_hip_y = right_coords[0, 1] # Accesing to coordinate y of right hip

        # Get shoulders coordinates
        left_shoulder_y = left_shoulder_coords[0, 1]
        right_shoulder_y = right_shoulder_coords[0, 1]

        # Check for confidences 
        '''
        Verifying that all coordinates have a higher confident 
        threhold reather than 'limit_conf
        '''''
        if (left_coords[:, 2] > limit_conf).all() and (right_coords[:, 2] > limit_conf).all() and \
           (left_hand_coords[:, 2] > limit_conf).all() and (right_hand_coords[:, 2] > limit_conf).all() and \
           (left_shoulder_coords[:, 2] > limit_conf).all() and (right_shoulder_coords[:, 2] > limit_conf).all():

            # Calculate the minimum angle in both legs
            angles = legs_angles(left_coords[:, :2], right_coords[:, :2]) # Using knee angle

            # Calculate progress based on knee angle
            min_angle = min(angles)
            '''
            Calculation of the squad progress.
             - Taking a reference angle (150), less the minimum angles, we are takig the progress.
             - The result is divided by 30 for representing that a 120 angle is a complet squad.
            '''
            PROGRESS = max(0, min(100, (150 - min_angle) / 30 * 100))

            if mode == 1:
                # Legs bent or stretched
                if (angles < 120).all() and STATE=='UP':
                    STATE = 'DOWN'
                elif (angles > 150).all() and STATE=='DOWN':
                    STATE = 'UP'

                # Update stack of states and count
                state_stack.append(STATE)
                if len(state_stack)==6:
                    if state_stack == deque(
                        ['DOWN', 'DOWN', 'DOWN', 'UP', 'UP', 'UP']):
                        COUNT += 1

            if mode == 2:
                # Check if hands position relative to hips
                hands_above_hips = (left_hand_coords[0, 1] < left_hip_y) and (right_hand_coords[0, 1] < right_hip_y)

                if hands_above_hips:
                    # Legs bent or stretched
                    if (angles < 120).all() and STATE == 'UP':
                        STATE = 'DOWN'
                    elif (angles > 150).all() and STATE == 'DOWN':
                        STATE = 'UP'

                    # Update stack of states and count
                    state_stack.append(STATE)
                    if len(state_stack) == 6:
                        if state_stack == deque(
                                ['DOWN', 'DOWN', 'DOWN', 'UP', 'UP', 'UP']):
                            COUNT += 1
                else:
                    HANDS_POSITION_WARNING = True
                    print("bad squatting")

            if mode == 3:
                    # Check hands position relative to shoulders
                hands_above_shoulders = (left_hand_coords[0, 1] < left_shoulder_y) and (right_hand_coords[0, 1] < right_shoulder_y)

                if hands_above_shoulders:
                    # Legs bent or stretched
                    if (angles < 120).all() and STATE == 'UP':
                        STATE = 'DOWN'
                    elif (angles > 150).all() and STATE == 'DOWN':
                        STATE = 'UP'

                    # Update stack of states and count
                    state_stack.append(STATE)
                    if len(state_stack) == 6:
                        if state_stack == deque(
                                ['DOWN', 'DOWN', 'DOWN', 'UP', 'UP', 'UP']):
                            COUNT += 1
                else:
                    HANDS_POSITION_WARNING = True
                    print("bad squatting")         

    # Show info if required
    if verbose:
        print(f"State: {STATE}")
        print(f"Count: {COUNT}")
        print(f"Progress: {PROGRESS:.0f}%")


def draw_grid_on_image(img, grid_size=(10, 10)):
    """
    Function to draw a grid on an image.
    """
    draw = ImageDraw.Draw(img)

    # Get image dimensions
    img_width, img_height = img.size

    # Calculate cell dimensions
    cell_width = img_width / grid_size[0]
    cell_height = img_height / grid_size[1]

    # Calculate vertical line positions
    vertical_lines = [(i * cell_width, 0, i * cell_width, img_height) for
                      i in range(grid_size[0] + 1)]

    # Calculate horizontal line positions
    horizontal_lines = [(0, i * cell_height, img_width, i * cell_height) for
                        i in range(grid_size[1] + 1)]

    # Draw all lines
    for line in vertical_lines + horizontal_lines:
        draw.line(line, fill="black")

    # Return the image with the grid
    return img


##################################################
# Main program
##################################################

use_camera = input('Do you ant to use camera (1) or a video (2)? ').strip().lower()

if use_camera == '1':
    source = 0  # Using camera
    mode_selected = True
else:
    source = None  # Using video, the video will be defined with the mode 
    mode_selected = False

# Prompt user to select mode
print('1 - Simple squat\n2 - Squats with hands above the hips\n3 - Squats with hands above the shoulders')
try:
    MODE = int(input('Input : '))
    if source is None:
        # Asignar la fuente del video según el modo seleccionado
        if MODE in [1, 2]:
            source = video_mode2_path
        elif MODE == 3:
            source = video_mode3_path
        else:
            print('Invalid mode, using mode 1 by default.')
            MODE = 1
            source = video_mode2_path
    elif MODE not in [1, 2, 3]:
        print('Invalid mode, using mode 1 by default.')
        MODE = 1
except ValueError:
    print('Not a number, using mode 1 by default.')
    MODE = 1
    if source is None:
        source = video_mode2_path

# Select source
cap = cv2.VideoCapture(source)
stream = True  # If stream=True the output is a generator
               # otherwise it's a list

# Loop through the video frames
cont = 0
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    # If the frame is empty, break the loop
    if not success:
        break

    # Perform pose estimation on this single frame
    results = model(source=frame,
                    show=True,
                    conf=0.85,  # Confidence greater than
                    save=False,
                    stream=stream)  # Create a generator instead of a list

    # Extract data from results
    if not stream:  # En caso de que stream=False
        r = results[0]
    else:
        r = next(results)

    if cont == 0:
        if ONE_IMAGE:
            cv2.destroyWindow('image0.jpg')
        else:
            cv2.setWindowTitle('image0.jpg', 'YoloV8 Results')

    # Convert to image
    if CHECK:
        im = draw_grid_on_image(Image.fromarray(r.plot()[..., ::-1]))
        #im = np.array[im]
        #im.show()

    evaluate_position(r,mode=MODE)

    frame_with_text = add_annotations(frame)

    # Display the annotated frame
    cv2.imshow('Squat Counter Window', frame_with_text)

    # Check for user input to break the loop (e.g., press 'q' to exit)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Increment frame counter
    cont +=1

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
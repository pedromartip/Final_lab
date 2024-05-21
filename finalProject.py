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

# Load the Yolov8 model
model = YOLO('src/models/yolov8s-pose.pt')

# Open the video file
source = 0
# source = "src/inference/videos/MySquats.mp4"

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

    # Define the position and font settings for the text
    text_position1 = (10, 30)
    text_position2 = (10, 60)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    green = (0, 255, 0)
    red = (0, 0, 255)
    font_color = green if STATE == 'UP' else red
    font_thickness = 2

    frame_with_text = frame.copy()
    cv2.putText(frame_with_text, state_text, text_position1, font,
                font_scale, font_color, font_thickness)
    cv2.putText(frame_with_text, count_text, text_position2, font,
                font_scale, green, font_thickness)

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
    left_hand = [


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
        # masks = result.masks

        # Probs object for classification outputs
        # probs = result.probs


def evaluate_position(result, limit_conf=0.3, verbose=False):
    """
    Evaluate position of the body in the image

    It updates the global variables STATE (UP or DOWN) and the number
    of squats done (COUNT)

    Args:
        result (Ultralytics Results): Results object from Ultralytics. It
            contains all the data of the pose estimation.
        limit_conf (float, optional): It's the limiting confidence. Greater
            confidences in (all) points estimation will be considered,
            otherwise they will be descarted. Defaults to 0.3.
        verbose (bool, optional): Print info. Defaults to False.
    """

    # Global variables
    global COUNT
    global STATE
    global state_stack

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
        left_coords, right_coords = get_legs_coords(kpts)

        # Check for confidences
        if (left_coords[:, 2] > limit_conf).all() and (right_coords[:, 2] > limit_conf).all():

            # Calculate the minimum angle in both legs
            angles = legs_angles(left_coords[:, :2], right_coords[:, :2])

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

    # Show info if required
    if verbose:
        print(f"State: {STATE}")
        print(f"Count: {COUNT}")


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
                    conf=0.3,  # Confidence greater than
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
    print('1 - Simple squad\n 2 - Squads with hands above the hips\n 3 - Squads with hands above the shoulders')
    try:
        mode = int(input('Input : '))
        if mode == 1:
            evaluate_position(r)
        elif mode == 2:
            evaluate_position_2(r)
        elif mode == 3:
            evaluate_position_2(r)

    except ValueError:
        print('Not a number')

    # Evaluate position
    evaluate_position(r)

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
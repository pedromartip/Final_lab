# -*- coding: utf-8 -*-

# License: GNU General Public License v3.0

# Some of these functions are copied or based on functions
# found on https://github.com/ricardodeazambuja/colab_utils
# under license GNU General Public License v3.0


#%%

####################################
# Libraries
####################################

import os
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET
from io import BytesIO
from base64 import b64encode, b64decode
import yaml
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from ultralytics.engine.results import Results
from IPython.display import display, Javascript, HTML

# Try-except necessary when testing locally
try:
    from google.colab.output import eval_js
    import ffmpeg
except ImportError:
    pass


#%%

####################################
# Functions
####################################

def draw_boxes(image, boxes):
    """
    Draw bounding boxes and labels on an image.

    Parameters:
    - image (PIL.Image.Image): The image on which to draw the boxes.
    - boxes (list of tuples): A list of tuples, each representing a
        bounding box and a label. Each tuple should have the format
        (xmin, ymin, xmax, ymax, label), where (xmin, ymin) is the
        top-left corner, and (xmax, ymax) is the bottom-right corner
        of the bounding box.

    Returns:
    - PIL.Image.Image: The image with drawn boxes and labels.
    """
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    # Draw each box and label
    # Each item in boxes_with_labels is a tuple (xmin, ymin, xmax, ymax, label)
    for box in boxes:
        xmin, ymin, xmax, ymax, label = box
        label = str(int(label)) if isinstance(label, float) else str(label)

        # Draw the rectangle (box)
        draw.rectangle(((xmin, ymin), (xmax, ymax)), outline="blue", width=2)

        # Draw the label
        text_size = draw.textsize(label, font=font)
        draw.rectangle(((xmin, ymin - text_size[1]), (xmin + text_size[0], ymin)), fill="blue")
        draw.text((xmin, ymin - text_size[1]), label, fill="white", font=font)

    return image


#%%

def read_darknet_annotations(annotation_file, image_width, image_height):
    """
    Read and parse annotation data from a file in Darknet YOLO format.

    Parameters:
    - annotation_file (str): The file path of the annotation file.
    - image_width (int): The width of the image corresponding to the
        annotations.
    - image_height (int): The height of the image corresponding to the
        annotations.

    Returns:
    - list of tuples: A list where each tuple represents a bounding box and
        its associated class ID. The tuple format is (xmin, ymin, xmax,
        ymax, class_id), where (xmin, ymin) is the top-left corner, and
        (xmax, ymax) is the bottom-right corner of the bounding box.
    """

    # Open and read the annotation file
    with open(annotation_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    boxes = []  # List to store parsed bounding box data
    for line in lines:
        # Parse the line: class ID, normalized center coordinates, width,
        # and height
        class_id, x_center_norm, y_center_norm, width_norm, height_norm = \
            map(float, line.split())

        # Convert normalized values to pixel coordinates
        x_center = x_center_norm * image_width
        y_center = y_center_norm * image_height
        width, height = width_norm * image_width, height_norm * image_height

        # Calculate top-left and bottom-right corners of the bounding box
        xmin = int(x_center - width / 2)
        ymin = int(y_center - height / 2)
        xmax = int(x_center + width / 2)
        ymax = int(y_center + height / 2)

        # Append the bounding box data to the list
        boxes.append((xmin, ymin, xmax, ymax, class_id))

    return boxes


#%%

def show(input_var, **kwargs):
    """
    Display an image based on the input variable provided.

    The function can handle different types of input such as a file path, a
    Path object, a list of Results objects, a single Results object, or a
    numpy array representing an image.

    Parameters:
    - input_var: The input variable to be displayed. This can be a string
        (file path), a Path object, a list of Results objects, a single
        Results object, or a numpy ndarray.

    Returns:
    - img (PIL): Displays and returns the image as a PIL object. If the input
        is not recognized or cannot be processed, the function returns None.
    """
    # Control variables
    is_path = False
    val = None

    # Get annotation file if it's passed as argument
    annotation_file = kwargs.get('annotation_file', '')

    # Get size of the image to display or default
    default_image_size = 8
    size = kwargs.get('size', default_image_size)

    # Get return option
    return_object = kwargs.get('return_object', None)

    # If the input is a string, convert it to a Path object
    if isinstance(input_var, str):
        val = Path(input_var)
        is_path = True

    # If the input is already a Path object, it's fine
    elif isinstance(input_var, Path):
        val = input_var
        is_path = True

    # If the input is a list, check its first element
    # If the first element is a Results object, return it
    elif isinstance(input_var, list):
        val = input_var[0] if isinstance(input_var[0], Results) else None

    # If the input is a Results object, return it
    elif isinstance(input_var, Results):
        val = input_var

    # If the input is a numpy array, assign it directly to img
    elif isinstance(input_var, np.ndarray):
        img = input_var

    # If the input is nothing from above, discard it
    else:
        # If none of the conditions match, return None
        val = None

    # Manage discards
    if val is None and not isinstance(input_var, np.ndarray):
        return None

    # Manage PIL and numpy arrays
    if is_path:
        # Load the image and mask using PIL
        img = Image.open(val)
    elif not isinstance(input_var, np.ndarray):
        # Get the image from results
        img = val.plot()[...,::-1]

    # Convert numpy arrays to PIL image
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)

    # Add annotations if required
    if annotation_file:
        w, h = img.size
        boxes = read_darknet_annotations(annotation_file, w, h)
        draw_boxes(img, boxes)

    # Show the image
    plt.figure(figsize=(size, size))
    plt.imshow(img)
    plt.axis('off')  # To turn off axis numbers and ticks
    plt.show()

    # Return image object if selected in kwargs
    if return_object == 'image':
        return_object = img

    return return_object


#%%

def load_classes_from_yaml(yaml_path):
    """
    Function to extract the names form the yaml file

    Args:
        yaml_path (path): The path to the yaml file

    Outputs:
        class2id (dict): Dictionary mapping class names (strings) with
            identifiers (integers)
        class2id (list): List of ordered class names, so this list maps
            the position (integer) with class names (strings)
    """
    with open(yaml_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
        id2class = data['names']

    id2class = list(data['names'].values())
    class2id = {v: int(k) for k, v in data['names'].items()}

    return id2class, class2id


#%%

def list_files_in_folder(folder_path, absolute=False, recursivity=False):
    """
    List all files in the specified folder and optionally in its subfolders
    recursively, including full absolute paths if specified.

    Parameters:
    - folder_path: Path to the folder.
    - absolute (bool, optional): If True, include the full absolute paths of
        the files. Default is False.
    - recursivity (bool, optional): If True, include files from subfolders
        recursively. Default is False.

    Returns:
    - List of file names in the folder (and subfolders if recursivity is
        True), with or without full paths based on the 'absolute' parameter.
    """
    files = []
    if recursivity:
        # Recursive case: walk through the directory and its subdirectories
        for root, dirs, filenames in os.walk(folder_path):
            for filename in filenames:
                full_path = os.path.join(root, filename)
                files.append(full_path if absolute else filename)
    else:
        # Non-recursive case: list entries in the specified folder
        all_entries = os.listdir(folder_path)
        for entry in all_entries:
            if os.path.isfile(os.path.join(folder_path, entry)):
                files.append(os.path.abspath(os.path.join(
                    folder_path, entry)) if absolute else entry)

    return files


#%%

def copy_files(files_list, destination_folder, overwrite=False):
    """
    Copies a list of files to a specified destination folder.

    This function iterates over a list of file paths, checks if each file
    exists, and if so, copies it to the specified destination folder. If the
    'overwrite' parameter is set to True, existing files in the destination
    will be overwritten. Otherwise, they will be skipped. If the destination
    folder does not exist, it will be created.

    Parameters:
    files_list (list of str): A list containing the absolute paths of files
        to be copied.
    destination_folder (str): The absolute path of the destination folder
        where files will be copied.
    overwrite (bool, optional): Whether to overwrite files in the destination
        folder. Default is False.

    Returns:
    None
    """

    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Copy each file from the list to the destination
    for file_path in files_list:
        # Extract the filename from the file path
        filename = os.path.basename(file_path)

        # Construct the full path of the file in the destination folder
        destination_file_path = os.path.join(destination_folder, filename)

        # Check if the file exists in the destination
        if os.path.isfile(file_path) and (overwrite or not os.path.isfile(
            destination_file_path)):
            shutil.copy(file_path, destination_folder)


#%%

def get_classes_from_voc(input_path):
    """
    Extract all unique class names from Pascal VOC XML annotations.
    
    Parameters:
    - input_path: Path to an XML file or a folder containing XML files.
    
    Returns:
    - List of unique class names.
    """

    # Initialize list of unique class names. We need a list and
    # not a set to keep track of the order
    class_names = []

    # List of XML files to process
    xml_files = []

    # Check if input_path is a folder or a single XML file
    if os.path.isdir(input_path):
        xml_files = [os.path.join(input_path, f) for f in os.listdir(
            input_path) if f.endswith('.xml')]
    else:
        xml_files.append(input_path)

    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Extract class names from each XML file and add to the list
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in class_names:
                class_names.append(class_name)

    return list(class_names)


#%%

def write_objects_to_file(root, output_txt, class_names, width, height):
    """
    Writes object data in YOLO format to the output text file.

    Parameters:
    - root: Root element of the XML tree.
    - output_txt: Path to the output text file.
    - class_names: List of class names.
    - width: Width of the image.
    - height: Height of the image.
    """
    with open(output_txt, 'w', encoding='utf-8') as txt_file:
        for obj in root.findall('object'):
            # Get class name and convert to class index
            class_name = obj.find('name').text
            class_idx = class_names.index(class_name)

            # Extract bounding box coordinates
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)

            # Convert VOC format (xmin, ymin, xmax, ymax) to YOLO format
            # (x_center, y_center, width, height)
            x_center = (xmin + xmax) / 2.0 / width
            y_center = (ymin + ymax) / 2.0 / height
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height

            # Write to txt file
            txt_file.write(f"{class_idx} {x_center} {y_center} {w} {h}\n")


#%%

def voc_to_yolo(input_path, output_folder, class_names):
    """
    Convert Pascal VOC format XML to YOLO Darknet format txt.
    
    Parameters:
    - input_path: Path to the XML file or folder containing multiple XML files
    - output_folder: Folder to save the output txt files.
    - class_names: List of class names written as in the XML files. Note that
        the order of this list fixes the id of each class.
    """

    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List of XML files to process
    xml_files = []

    # Check if input_path is a folder or a single XML file
    if os.path.isdir(input_path):
        xml_files = [os.path.join(input_path, f) for f in os.listdir(
            input_path) if f.endswith('.xml')]
    else:
        xml_files.append(input_path)

    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Extract image dimensions
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        output_txt = os.path.join(output_folder,
            os.path.basename(xml_file).replace('.xml', '.txt'))

        # Call the new function to write object data
        write_objects_to_file(root, output_txt, class_names, width, height)


#%%

############################################################################
# Functions imported from https://github.com/ricardodeazambuja/colab_utils
############################################################################

def labelImage(inputImg, imgformat='PNG', deleteAfter=True, scale=1.0,
               line_color="green"):
    """
    Opens an image, records mouse clicks (boxes) and labels.

    Parameters:
    - inputImg: The input image (can be a path, numpy array, or PIL image).
    - imgformat: Format of the image ('PNG', 'JPEG', etc.).
    - deleteAfter: Boolean indicating if the image should be deleted after
        labeling.
    - scale: Scaling factor for the image display.
    - line_color: Color of the bounding box.

    Returns:
    - list: [box (list), label (str)]
    """

    # JavaScript code for handling image labeling in the browser
    JS_SRC = """
    async function label_image(scale) {
        const image  = document.getElementById("inputImage");
        const w = image.width;
        const h = image.height;

        const image_div = document.getElementById("image_div");

        // Create interface buttons and textbox
        const ok_btn = document.createElement('button');
        ok_btn.textContent = 'Finish';
        const add_btn = document.createElement('button');
        add_btn.textContent = 'Add';
        const clr_btn = document.createElement('button');
        clr_btn.textContent = 'Clear';
        const textbox = document.createElement('input');
        textbox.textContent = "text";

        const canvas = document.createElement('canvas');
        canvas.width = w;
        canvas.height = h;

        var ctx = canvas.getContext('2d');
        canvas.style.position = 'absolute';
        canvas.style.left = '0px';
        canvas.style.top = '0px';
        canvas.style.z_index = 1000;
        canvas.style.border = 0;
        canvas.style.padding = 0;
        canvas.style.margin = 0;

        image_div.appendChild(canvas);

        const interface_div = document.getElementById("interface_div");
        interface_div.appendChild(textbox);
        interface_div.appendChild(add_btn);
        interface_div.appendChild(ok_btn);
        interface_div.appendChild(clr_btn);

        textbox.width = 100;

        var x1,x2,y1,y2;
        var clickNumber = 0;
        var boxes = new Array();
        var try_again = true;

        while (try_again){
            await new Promise((resolve) => {
                canvas.onclick = () => {
                    console.log("X:"+event.clientX+" Y:"+event.clientY); 
                    if(clickNumber==0){
                        x1 = event.clientX;
                        y1 = event.clientY;
                        clickNumber = 1;
                    } else if(clickNumber==1){
                        x2 = event.clientX;
                        y2 = event.clientY;
                        ctx.lineWidth = 5;
                        ctx.strokeStyle = '%s';
                        ctx.strokeRect(x1, y1, x2-x1, y2-y1);
                        clickNumber = 2;
                    }
                    resolve();
                };
                ok_btn.onclick = () => {
                    try_again=false; 
                    boxes.push([[x1/w, y1/h, (x2-x1)/w, (y2-y1)/h],textbox.value]);
                    if("%s" == "True"){
                        tmp_div = document.getElementById("main_div");
                        tmp_div.remove();
                    }
                    resolve();
                };
                add_btn.onclick = () => {
                    if (clickNumber==2){
                        boxes.push([[x1/w, y1/h, (x2-x1)/w, (y2-y1)/h],textbox.value]);
                        clickNumber = 0;
                    }
                    resolve();
                };
                clr_btn.onclick = () => { 
                    ctx.clearRect(0, 0, canvas.width, canvas.height); 
                    boxes = new Array();
                    clickNumber = 0;
                    resolve();
                };
            });
        }
        return boxes;
    }
    """ % (line_color, str(deleteAfter))

    imageBuffer = BytesIO()

    # Handle different types of input images
    if type(inputImg) == str:
        img = Image.open(inputImg)
        w, h = img.size
        img.save(imageBuffer, format=imgformat)
    elif type(inputImg) == np.ndarray:
        img = Image.fromarray(inputImg)
        w, h = img.size
        img.save(imageBuffer, format=imgformat)
    elif "PIL" in str(type(inputImg)):
        w, h, _ = inputImg.size
        inputImg.save(imageBuffer, format=imgformat)

    imgBase64 = b64encode(imageBuffer.getvalue())

    # Convert image to base64 for embedding in HTML
    if imgformat == 'PNG':
        str_data = "data:image/png;base64," + imgBase64.decode(encoding="utf-8")
    elif imgformat == 'JPEG' or imgformat == 'JPG':
        str_data = "data:image/jpeg;base64," + imgBase64.decode(encoding="utf-8")
    elif imgformat == 'GIF':
        str_data = "data:image/gif;base64," + imgBase64.decode(encoding="utf-8")
    else:
        raise "Wrong image format!"

    # HTML code for displaying the image and interface
    HTML_SRC = f"""
    <div id="main_div" style="padding:0; margin:0; border:0; height:{h*scale+50}px; width:{w*scale}px;">
    <div id="image_div" style="padding:0; margin:0; border:0; height:{h*scale}px; width:{w*scale}px; position:absolute; top:0px; left:0px;">
    <img id="inputImage" src="{str_data}" style="padding:0; margin:0; border:0; position:absolute; top:0px; left:0px;" height={h*scale}px; width={w*scale}px;/>
    </div>
    <div id="interface_div" style="padding:0; margin:0; border:0; position:absolute; top:{h*scale}px; left:0px;"></div>
    </div>
    """
    display(HTML(HTML_SRC))
    display(Javascript(JS_SRC))
    data = eval_js(f'label_image({float(scale)})')
    return data


#%%

def webcam2numpy(quality=0.8, size=(800,600)):
    """
    Saves images from your webcam into a numpy array.

    Parameters:
    - quality: Quality of the saved image (0.0 to 1.0).
    - size: Tuple indicating the width and height of the webcam feed.

    Returns:
    - numpy.ndarray: Image captured from the webcam as a numpy array.
    """

    # HTML and JavaScript code to access and display the webcam feed
    VIDEO_HTML = """
    <div class="video_container">
        <video autoplay width=%d height=%d></video>
        <div style='position: absolute;top: 40px; left: 40px; font-size: 40px; color: green;'>Click on the image to save!</div>
    </div>

    <script>
        var video = document.querySelector('video')

        // Access the webcam feed
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => video.srcObject = stream)

        var data = new Promise(resolve => {
            video.onclick = () => {
                var canvas = document.createElement('canvas')
                var [w, h] = [video.offsetWidth, video.offsetHeight]
                canvas.width = w
                canvas.height = h
                canvas.getContext('2d')
                    .drawImage(video, 0, 0, w, h)
                // Stop the webcam feed
                video.srcObject.getVideoTracks()[0].stop()
                // Replace the video element with the canvas
                video.replaceWith(canvas)
                // Resolve the promise with the image data
                resolve(canvas.toDataURL('image/jpeg', %f))
            }
        })
    </script>
    """

    # Display the webcam feed using the HTML and JavaScript code
    handle = display(HTML(VIDEO_HTML % (size[0], size[1], quality)),
                     display_id='videoHTML')

    # Get the image data from the JavaScript
    data = eval_js("data")

    # Convert the base64 encoded data to binary
    binary = b64decode(data.split(',')[1])

    # Convert the binary data to a numpy array
    f = BytesIO(binary)
    return np.asarray(Image.open(f))

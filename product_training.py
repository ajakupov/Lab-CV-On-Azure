# coding utf-8
import cv2
import os
import tensorflow as tf
import numpy as np
from numba import jit

# BGR colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
# Output text parameters
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
LINE_TYPE = 1

WINDOW_NAME = 'product trainer'


def resize_down_to_1600_max_dim(image):
    """Change oversized image dimensions using Linear Interpolation

    Arguments:
        image {OpenCV} -- OpenCV image

    Returns:
        OpenCV -- resized or initial image
    """
    h, w = image.shape[:2]
    if (h < 1600 and w < 1600):
        return image

    new_size = (1600 * w // h, 1600) if (h > w) else (1600, 1600 * h // w)
    return cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)


def crop_center(img, cropx, cropy):
    """Extract a middle part of an image

    Arguments:
        img {OpenCv} -- OpenCV image to be cropped
        cropx {[type]} -- width of the cropped region
        cropy {[type]} -- height of the cropped region

    Returns:
        [OpenCV] -- cropped image
    """
    h, w = img.shape[:2]
    startx = w//2-(cropx//2)
    starty = h//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx]


def resize_to_256_square(image):
    """Resize an image using the Linear Interpolation

    Arguments:
        image {OpenCV} -- OpenCV image

    Returns:
        OpenCV -- resized image
    """
    h, w = image.shape[:2]
    return cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)


def save_image(image, folder):
    """Save an image with unique name

    Arguments:
        image {OpanCV} -- image object to be saved
        folder {string} -- output folder
    """

    # check whether the folder exists and create one if not
    if not os.path.exists(folder):
        os.makedirs(folder)

    # to not erase previously saved photos counter (image name) = number of photos in a folder + 1
    image_counter = len([name for name in os.listdir(folder)
                         if os.path.isfile(os.path.join(folder, name))])

    # increment image counter
    image_counter += 1

    # save image to the dedicated folder (folder name = label)
    cv2.imwrite(folder + '/' + str(image_counter) + '.png', image)


# graph of operations to upload trained model
graph_def = tf.compat.v1.GraphDef()
# list of classes
labels = ['activia', 'veloute']


# N.B. Azure Custom vision allows export trained model in the form of 2 files
# model.pb: a tensor flow graph and labels.txt: a list of classes
# import tensor flow graph, r+b mode is open the binary file in read or write mode
with tf.io.gfile.GFile(name='product_model.pb', mode='rb') as f:
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def=graph_def, name='')

# initialize video capture object to read video from external webcam
video_capture = cv2.VideoCapture(1)
# if there is no external camera then take the built-in camera
if not video_capture.read()[0]:
    video_capture = cv2.VideoCapture(0)

# Full screen mode
cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(
    WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
# These names are part of the model and cannot be changed.
output_layer = 'loss:0'
input_node = 'Placeholder:0'
predicted_tag = 'Predicted Tag'

# counter to control the percentage of saved images
frame_counter = 0

with tf.compat.v1.Session() as sess:
    prob_tensor = sess.graph.get_tensor_by_name(output_layer)
    while(video_capture.isOpened()):
        # read video frame by frame
        ret, frame = video_capture.read()

        try:
            frame = cv2.flip(frame, 1)
            frame_counter += 1
            # frame width and height
            w, h = 200, 300
            # set upper and lower boundaries
            upX = 220
            upY = 50
            lowX = upX + w
            lowY = upY + h
            image = frame[upY:lowY, upX:lowX]

            # If the image has either w or h greater than 1600 we resize it down respecting
            # aspect ratio such that the largest dimension is 1600
            image = resize_down_to_1600_max_dim(image)

            # We next get the largest center square
            h, w = image.shape[:2]
            min_dim = min(w, h)
            max_square_image = crop_center(image, min_dim, min_dim)

            # Resize that square down to 256x256
            augmented_image = resize_to_256_square(image)

            # Get the input size of the model
            input_tensor_shape = sess.graph.get_tensor_by_name(
                input_node).shape.as_list()
            network_input_size = input_tensor_shape[1]

            # Crop the center for the specified network_input_Size
            augmented_image = cv2.resize(
                image, (network_input_size, network_input_size), interpolation=cv2.INTER_LINEAR)

            predictions = sess.run(
                prob_tensor, {input_node: [augmented_image]})

            # get the highest probability label
            highest_probability_index = np.argmax(predictions)
            predicted_tag = labels[highest_probability_index]
            output_text = predicted_tag
            if predicted_tag == 'activia':
                frameColor = GREEN
            elif predicted_tag == 'veloute':
                frameColor = BLUE
            else:
                frameColor = RED

            cv2.rectangle(frame, (upX, upY), (lowX, lowY), frameColor, 1)

            if (frame_counter % 10 == 0):
                save_image(augmented_image, predicted_tag)

        except:
            continue
        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# release video capture object
video_capture.release()
cv2.destroyAllWindows()

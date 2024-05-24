# from PIL import Image
# import matplotlib.pyplot as plt

import os
# import pickle
import numpy as np
# from tqdm.notebook import tqdm
from datetime import datetime 
# from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.models import Model
# from tensorflow.keras.utils import to_categorical, plot_model
# from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
import pyttsx3
engine = pyttsx3.init()
engine.setProperty('rate', 110)     # Speed percent (can go over 100)
import cv2
import numpy as np

class YoloDetector:
    def __init__(self, config_path='archive/yolov3.cfg', weights_path='archive/yolov3.weights', class_path='archive/coco.names'):
        self.net = cv2.dnn.readNet(weights_path, config_path)
        self.classes = []
        with open(class_path, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.layer_names = self.net.getLayerNames()
        self.output_layers =  self.net.getUnconnectedOutLayersNames()
        
    def detect_objects(self, image):
        height, width,channels = image.shape
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        detected_objects = []
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                detected_objects.append({'label': label, 'box': (x, y, w, h)})

        return detected_objects

# Load YOLO model
yolo = YoloDetector()

# set path in which you want to save images 
path = r"E:\\1 Image caption generator\\5 Image Caption\\archive\\frames"

# changing directory to given path 
os.chdir(path) 

# i variable is to give unique name to images 
i = 1

wait = 0

# Open the camera 
video = cv2.VideoCapture(0) 

def find_repeated_elements(array):
    repeated_elements = {}

  # Iterate over the array.
    for element in array:

        # If the element is not already in the dictionary, add it with a count of 1.
        if element not in repeated_elements:
            repeated_elements[element] = 1

        # If the element is already in the dictionary, increment its count.
        else:
            repeated_elements[element] += 1

    # Return the dictionary of repeated elements.
    return repeated_elements

while True: 
    # Read video by read() function and it 
    # will extract and return the frame 
    ret, img = video.read() 

    # Put current DateTime on each frame 
    font = cv2.FONT_HERSHEY_PLAIN 
    cv2.putText(img, str(datetime.now()), (20, 40), 
                font, 2, (255, 255, 255), 2, cv2.LINE_AA) 

    # Display the image 
    cv2.imshow('live video', img) 

    # wait for user to press any key 
    key = cv2.waitKey(100) 

    # wait variable is to calculate waiting time 
    wait = wait + 100

    if key == ord('q'): 
        break
    # when it reaches to 5000 milliseconds 
    # we will save that frame in given folder 
    if key == ord('g'): 
        filename = 'Frame_'+str(i)+'.jpg'
        
        # Save the images in given path 
        cv2.imwrite(filename, img) 
        i = i + 1
        wait = 0
        # model starts here
        # image_path = "archive\\frames\\Frame_"+str(i-1)+".jpg"
        image_path = os.path.join(path, 'Frame_' + str(i-1) + '.jpg')
        
        # load image
        image = load_img(image_path, target_size=(224, 224))
        # convert image pixels to numpy array
        image = img_to_array(image)
        # reshape data for model
        # image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        
        
        # # Load image
        # image = cv2.imread('archive\\Images\\10815824_2997e03d76.jpg')

        # Perform object detection
        objects = yolo.detect_objects(image)

        # Extract detected objects and their labels
        detected_objects = [obj['label'] for obj in objects]
        bounding_boxes = [obj['box'] for obj in objects]
        frequency_object = find_repeated_elements(detected_objects)


        if(detected_objects==[]):
            caption="nothing in front of you."
        
        elif(len(detected_objects)==1):
            for i, obj in enumerate(detected_objects):
                caption=obj+ " in front of you."
        else:
            caption = ""
            keys = list(frequency_object.keys())
            for j in keys:
                caption += str(frequency_object[j])+" "+ j+ " and "
            caption=caption[:-4]+ "are in front of you."

        print("Caption:", caption)
        a=caption
        # mytext = a
        engine.say(a)
        engine.runAndWait()

# close the camera 
video.release() 

# close open windows 
cv2.destroyAllWindows() 

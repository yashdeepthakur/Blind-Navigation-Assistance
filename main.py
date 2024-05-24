import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from datetime import datetime
import os
import numpy as np
from datetime import datetime 
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pyttsx3
engine = pyttsx3.init()
engine.setProperty('rate', 110)     # Speed percent (can go over 100)
import numpy as np
# Paste YoloDetector class and necessary imports here
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

class App:
    def __init__(self, window, video_source=0):
        self.window = window
        self.window.title("Live Video Captioning")

        # Open the camera
        self.video_source = video_source
        self.vid = cv2.VideoCapture(self.video_source)

        # Set up YOLO model
        self.yolo = YoloDetector()

        # Create canvas to display video
        self.canvas = tk.Canvas(window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        # Button to capture frame
        self.btn_capture = tk.Button(window, text="Capture Frame", command=self.capture_frame)
        self.btn_capture.pack()

        # Label to display caption
        self.caption_label = tk.Label(window, text="")
        self.caption_label.pack()

        self.delay = 10
        self.update()

        self.window.mainloop()

    def capture_frame(self):
        ret, frame = self.vid.read()
        if ret:
            filename = 'Frame_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.jpg'
            cv2.imwrite(filename, frame)
            messagebox.showinfo("Capture", "Frame captured successfully!")

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            # Perform object detection
            objects = self.yolo.detect_objects(frame)

            # Extract detected objects and their labels
            detected_objects = [obj['label'] for obj in objects]
            bounding_boxes = [obj['box'] for obj in objects]

            if not detected_objects:
                caption = "Nothing in front of you."
            elif len(detected_objects) == 1:
                caption = detected_objects[0] + " in front of you."
            else:
                frequency_object = find_repeated_elements(detected_objects)
                caption = ""
                keys = list(frequency_object.keys())
                for j in keys:
                    caption += str(frequency_object[j]) + " " + j + " and "
                caption = caption[:-4] + " are in front of you."

            # Update caption label
            self.caption_label.config(text="Caption: " + caption)

            # Display video stream on canvas
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)


def main():
    # set path in which you want to save images
    path = r"E:\\1 Image caption generator\\5 Image Caption\\archive\\frames"
    # changing directory to given path
    os.chdir(path)

    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()

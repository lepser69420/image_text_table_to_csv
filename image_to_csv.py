from tkinter import Tk, filedialog
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from paddleocr import PaddleOCR, draw_ocr
import cv2
import os
import numpy as np
import tensorflow as tf

import pandas as pd

def select_image():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    image_path = file_path
    root.destroy()
    if file_path:

        print("Selected file:", file_path)
    return file_path

def convert(imagepath):
    ocr = PaddleOCR(use_angle_cls=True, lang='en')

    extracted_text = []

    # Load image, grayscale, Otsu's threshold
    normalized_path = os.path.normpath(imagepath)
    image = cv2.imread(f'{normalized_path}')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Repair horizontal table lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Remove horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (55, 2))
    detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(image, [c], -1, (255, 255, 255), 9)

    # Remove vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 55))
    detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(image, [c], -1, (255, 255, 255), 9)

    # Perform OCR
    text = ocr.ocr(image, cls=True)[0]
    boxes = [line[0] for line in text]
    texts = [line[1][0] for line in text]
    probabilities = [line[1][1] for line in text]
    image_boxes = image.copy()
    for box, text in zip(boxes, texts):
        cv2.rectangle(image_boxes, (int(box[0][0]), int(box[0][1])), (int(box[2][0]), int(box[2][1])), (0, 0, 255),
                      1)
        cv2.putText(image_boxes, text, (int(box[0][0]), int(box[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (222, 0, 0),
                    1)
    im = image.copy()
    image_height = image.shape[0]
    image_width = image.shape[1]
    horiz_boxes = []
    vert_boxes = []

    for box in boxes:
        x_h, x_v = 0, int(box[0][0])
        y_h, y_v = int(box[0][1]), 0
        width_h, width_v = image_width, int(box[2][0] - box[0][0])
        height_h, height_v = int(box[2][1] - box[0][1]), image_height

        horiz_boxes.append([x_h, y_h, x_h + width_h, y_h + height_h])
        vert_boxes.append([x_v, y_v, x_v + width_v, y_v + height_v])

        cv2.rectangle(im, (x_h, y_h), (x_h + width_h, y_h + height_h), (0, 0, 255), 1)
        cv2.rectangle(im, (x_v, y_v), (x_v + width_v, y_v + height_v), (0, 255, 0), 1)
    horiz_out = tf.image.non_max_suppression(
        horiz_boxes,
        probabilities,
        max_output_size=1000,
        iou_threshold=0.1,
        score_threshold=float('-inf'),
        name=None
    )
    horiz_lines = np.sort(np.array(horiz_out))
    im_nms = image.copy()
    for val in horiz_lines:
        cv2.rectangle(im_nms, (int(horiz_boxes[val][0]), int(horiz_boxes[val][1])),
                      (int(horiz_boxes[val][2]), int(horiz_boxes[val][3])), (0, 0, 255), 1)
    vert_out = tf.image.non_max_suppression(
        vert_boxes,
        probabilities,
        max_output_size=1000,
        iou_threshold=0.1,
        score_threshold=float('-inf'),
        name=None
    )
    vert_lines = np.sort(np.array(vert_out))
    for val in vert_lines:
        cv2.rectangle(im_nms, (int(vert_boxes[val][0]), int(vert_boxes[val][1])),
                      (int(vert_boxes[val][2]), int(vert_boxes[val][3])), (255, 0, 0), 1)
    out_array = [["" for i in range(len(vert_lines))] for j in range(len(horiz_lines))]
    unordered_boxes = []

    for i in vert_lines:
        unordered_boxes.append(vert_boxes[i][0])
    ordered_boxes = np.argsort(unordered_boxes)

    def intersection(box_1, box_2):
        return [box_2[0], box_1[1], box_2[2], box_1[3]]

    def iou(box_1, box_2):
        x_1 = max(box_1[0], box_2[0])
        y_1 = max(box_1[1], box_2[1])
        x_2 = min(box_1[2], box_2[2])
        y_2 = min(box_1[3], box_2[3])

        inter = abs(max((x_2 - x_1, 0)) * max((y_2 - y_1), 0))
        if inter == 0:
            return 0

        box_1_area = abs((box_1[2] - box_1[0]) * (box_1[3] - box_1[1]))
        box_2_area = abs((box_2[2] - box_2[0]) * (box_2[3] - box_2[1]))

        return inter / float(box_1_area + box_2_area - inter)

    for i in range(len(horiz_lines)):
        for j in range(len(vert_lines)):
            resultant = intersection(horiz_boxes[horiz_lines[i]], vert_boxes[vert_lines[ordered_boxes[j]]])

            for b in range(len(boxes)):
                the_box = [boxes[b][0][0], boxes[b][0][1], boxes[b][2][0], boxes[b][2][1]]
                if (iou(resultant, the_box) > 0.1):
                    out_array[i][j] = texts[b]
    out_array = np.array(out_array)
    pd.DataFrame(out_array).to_csv('output.csv')

class ImageToCSVConverter(BoxLayout):
    image_path = str()
    def __init__(self, **kwargs):
        super(ImageToCSVConverter, self).__init__(**kwargs)
        self.orientation = "vertical"

        self.convert_button = Button(on_press=self.convert_image_to_csv, text="Convert to CSV", size_hint=(None, None), width=200, height=50)
        self.convert_button.bind(on_press=self.convert_image_to_csv)
        self.add_widget(self.convert_button)

        self.converted_file_label = Label(text="")
        self.add_widget(self.converted_file_label)

    def select_image(self, instance):
        pass

    def convert_image_to_csv(self, instance):

        convert(select_image())

class ImageToCSVApp(App):
    def build(self):
        return ImageToCSVConverter()

if __name__ == "__main__":
    ImageToCSVApp().run()

from tkinter import filedialog, Tk
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.dropdown import DropDown
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.gridlayout import GridLayout
from kivy.lang import Builder
from paddleocr import PaddleOCR, draw_ocr # main OCR dependencies
import cv2 #opencv
import os # folder directory navigation
from kivy.uix.spinner import Spinner

def select_image():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    root.destroy()
    if file_path:
        print("Selected file:", file_path)
        return file_path

def detect(file_path, lang):
    ocr_model = PaddleOCR(lang= lang)
    normalized_path = os.path.normpath(file_path)
    print(normalized_path,lang)
    image_path = cv2.imread(normalized_path)
    result = ocr_model.ocr(image_path)
    for i in result:
        print(i[0][1])


class ImageSelector(BoxLayout):
    image_path = str()
    def __init__(self, **kwargs):
        super(ImageSelector, self).__init__(**kwargs)
        self.orientation = "vertical"
        self.padding = [50, 50, 50, 50]
        self.spacing = 20
        self.size_hint_y = None
        self.height = 600

        self.dropdown_layout = GridLayout(cols=2, size_hint=(1, 0.1))
        self.add_widget(self.dropdown_layout)

        self.select_language_label = Label(text="Select Language:",  size_hint=(None, None), width=200, height=50)
        self.dropdown_layout.add_widget(self.select_language_label)
        dropdown_values = ['en', 'np']

        self.spinner = Spinner(text='Select Option', values=dropdown_values, size_hint=(None, None), size=(100, 44))

        # Bind the on_text event to a function
        self.spinner.bind(on_text=self.detect_text)

        self.add_widget(self.spinner)

        self.detect_text_button = Button(text="Detect Text", size_hint=(None, None), width=200, height=50)
        self.detect_text_button.bind(on_press=self.detect_text)
        self.add_widget(self.detect_text_button)

        self.selected_language_label = Label(text="",  size_hint=(None, None), width=200, height=50)
        self.add_widget(self.selected_language_label)

    def detect_text(self, instance):
        detect(select_image(),self.spinner.text)

    def on_select_language(self, instance, language):
        self.selected_language_label.text = f"Selected Language: {language}"

class ImageSelectionApp(App):
    def build(self):
        return ImageSelector()

if __name__ == "__main__":
    ImageSelectionApp().run()



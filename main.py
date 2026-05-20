import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import time
import numpy as np
import threading
import pyttsx3
from collections import Counter
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Translator")
        self.root.geometry("800x600")

        self.video_label = ttk.Label(root)
        self.video_label.pack()

        self.word_var = tk.StringVar()
        self.word_label = ttk.Label(root, textvariable=self.word_var, font=("Helvetica", 16))
        self.word_label.pack(pady=10)

        button_frame = ttk.Frame(root)
        button_frame.pack()

        self.start_button = ttk.Button(button_frame, text="Start Camera", command=self.start_camera)
        self.start_button.grid(row=0, column=0, padx=5)

        self.stop_button = ttk.Button(button_frame, text="Stop Camera", command=self.stop_camera)
        self.stop_button.grid(row=0, column=1, padx=5)

        self.speak_button = ttk.Button(button_frame, text="Speak Word", command=self.speak_word)
        self.speak_button.grid(row=0, column=2, padx=5)

        self.clear_button = ttk.Button(button_frame, text="Clear Word", command=self.clear_word)
        self.clear_button.grid(row=0, column=3, padx=5)

        self.cap = None
        self.running = False
        self.current_word = ""
        self.prediction_buffer = []
        self.buffer_size = 15
        self.last_prediction_time = 0
        self.last_delete_time = 0

        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)

        self.IMAGE_SIZE = 200
        self.CROP_SIZE = 200

        self.data_generator = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)

        self.model = load_model('D:/Sign_language/asl_alphabet_9575.h5')

        with open("classes.txt", 'r') as f:
            self.classes = f.readline().split()
        self.classes.sort()

    def start_camera(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            self.running = True
            threading.Thread(target=self.video_loop).start()

    def stop_camera(self):
        if self.running:
            self.running = False
            self.cap.release()

    def clear_word(self):
        self.current_word = ""
        self.word_var.set(self.current_word)

    def speak_word(self):
        if self.current_word.strip():
            self.engine.say(self.current_word)
            self.engine.runAndWait()

    def video_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            cv2.rectangle(frame, (0, 0), (self.CROP_SIZE, self.CROP_SIZE), (0, 255, 0), 3)

            cropped_image = frame[0:self.CROP_SIZE, 0:self.CROP_SIZE]
            resized_frame = cv2.resize(cropped_image, (self.IMAGE_SIZE, self.IMAGE_SIZE))
            reshaped_frame = np.expand_dims(resized_frame, axis=0)
            frame_for_model = self.data_generator.standardize(np.float64(reshaped_frame))

            prediction = self.model.predict(frame_for_model, verbose=0)
            predicted_index = np.argmax(prediction)
            predicted_class = self.classes[predicted_index]
            prediction_probability = prediction[0, predicted_index]

            if prediction_probability > 0.5:
                self.prediction_buffer.append(predicted_class)
                if len(self.prediction_buffer) > self.buffer_size:
                    self.prediction_buffer.pop(0)
            else:
                self.prediction_buffer.clear()

            if self.prediction_buffer:
                smoothed_prediction = Counter(self.prediction_buffer).most_common(1)[0][0]
            else:
                smoothed_prediction = None

            current_time = time.time()

            if smoothed_prediction:
                if smoothed_prediction.lower() == 'del' and (current_time - self.last_delete_time) > 1.0:
                    if self.current_word:
                        self.current_word = self.current_word[:-1]
                    self.last_delete_time = current_time
                    self.prediction_buffer.clear()

                elif smoothed_prediction.lower() == 'space' and (current_time - self.last_prediction_time) > 2.5:
                    self.current_word += " "
                    self.prediction_buffer.clear()
                    self.last_prediction_time = current_time

                elif smoothed_prediction.isalpha() and smoothed_prediction.lower() not in ['nothing', 'del', 'space'] and (current_time - self.last_prediction_time) > 2.5:
                    self.current_word += smoothed_prediction
                    self.prediction_buffer.clear()
                    self.last_prediction_time = current_time

            self.word_var.set(f'Word: {self.current_word}')

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.mainloop()

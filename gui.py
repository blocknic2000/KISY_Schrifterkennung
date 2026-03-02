import tkinter as tk
from tkinter import Button
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class LetterRecognitionGUI:

    def __init__(self, root):
        self.root = root
        self.root.title("Buchstaben Erkennung")

        # Modell laden
        self.model = tf.keras.models.load_model("letter_recognition_model.h5")

        #                                  (A-Z)
        self.classes = [chr(i) for i in range(65, 91)]

        self.canvas_size = 300
        self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size, bg="white")
        self.canvas.grid(row=0, column=0, padx=10, pady=10)

        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 255)
        self.draw = ImageDraw.Draw(self.image)

        # use motion binding to draw smoothly
        self.canvas.bind("<B1-Motion>", self.draw_lines)
        self.canvas.bind("<ButtonPress-1>", self.start_draw)
        self.clear_button = Button(root, text="Löschen", command=self.clear_canvas)
        self.clear_button.grid(row=1, column=0)
        self.figure, self.ax = plt.subplots(figsize=(5, 4))
        self.bars = self.ax.bar(self.classes, [0]*26)
        self.ax.set_ylim(0, 1)
        self.ax.set_title("Wahrscheinlichkeit")
        self.ax.set_ylabel("Probability")
        self.chart = FigureCanvasTkAgg(self.figure, root)
        self.chart.get_tk_widget().grid(row=0, column=1, rowspan=2)
        self.update_prediction()

    def start_draw(self, event):
        self.last_x, self.last_y = event.x, event.y

    def draw_lines(self, event):
        x, y = event.x, event.y
        width = 10
        self.canvas.create_line(self.last_x, self.last_y, x, y,
                                fill="black", width=width, capstyle=tk.ROUND,
                                smooth=True)
        self.draw.line([self.last_x, self.last_y, x, y], fill=0, width=width)
        self.last_x, self.last_y = x, y

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 255)
        self.draw = ImageDraw.Draw(self.image)

    def preprocess(self):
        img = self.image.resize((32, 32))
        img_array = np.array(img).astype("float32")
        img_array = img_array / 255.0
        img_array = 1 - img_array
        img_array = img_array.flatten()
        img_array = img_array.reshape(1, 1024)
        return img_array
    def update_prediction(self):
        img_array = self.preprocess()
        predictions = self.model.predict(img_array, verbose=0)[0]
        for bar, prob in zip(self.bars, predictions):
            bar.set_height(prob)
        self.chart.draw()
        self.root.after(10, self.update_prediction)


if __name__ == "__main__":
    root = tk.Tk()
    app = LetterRecognitionGUI(root)
    root.mainloop()
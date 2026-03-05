from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import os

class LetterTrainer:

    def __init__(self):
        self.letters_dir = "server_letters"
        self.model_path = "letter_recognition_model.h5"
        self.img_size = 32
        self.num_classes = 26

    def load_data(self):
        images = []
        labels = []

        for letter_folder in sorted(os.listdir(self.letters_dir)):
            letter_path = os.path.join(self.letters_dir, letter_folder)

            if os.path.isdir(letter_path):
                for img_file in os.listdir(letter_path):
                    img_path = os.path.join(letter_path, img_file)
                    img = Image.open(img_path).convert("L")
                    img = img.resize((self.img_size, self.img_size))
                    img_array = np.array(img).astype("float32")
                    img_array = img_array / 255.0
                    img_array = 1 - img_array
                    img_array = img_array.flatten()
                    images.append(img_array)
                    labels.append(ord(letter_folder.upper()) - ord("A"))

        X = np.array(images)
        y = np.array(labels)

        print(f"Loaded {len(X)} images")
        return X, y

    def build_model(self):
        model = Sequential([
            Input(shape=(1024,)),
            Dense(256, activation='relu'),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(self.num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train(self):
        X, y = self.load_data()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.1,
            random_state=42,
            shuffle=True
        )

        model = self.build_model()

        model.fit(
            X_train,
            y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2
        )

        loss, accuracy = model.evaluate(X_test, y_test)
        print(f"\nTest Accuracy: {accuracy:.2%}")

        model.save(self.model_path)
        print(f"Model saved as '{self.model_path}'")


if __name__ == "__main__":
    trainer = LetterTrainer()
    trainer.train()
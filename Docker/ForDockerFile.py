import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import random
from tensorflow.keras import models, layers, optimizers
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import ImageTk, Image
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import random

# Load and preprocess the image
def load_and_preprocess_image(path):
    image = cv2.imread(path)  # Example: Using OpenCV to load the image
    image = cv2.resize(image, (150, 150))  # Example: Resize the image to (150, 150)
    image = image.astype(np.float32)  # Example: Convert the image to float32
    image /= 255.0  # Example: Normalize the image by dividing by 255.0
    return image

# Add Rain Effect
def add_rain_effect(image, raindrops=random.randint(200, 300), thickness=1, length=20, angle=10):
    height, width, _ = image.shape

    for _ in range(raindrops):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        l = np.random.randint(1, length)
        theta = np.random.randint(80, 100)

        dx = int(l * np.cos(np.deg2rad(theta)))
        dy = int(l * np.sin(np.deg2rad(theta)))

        cv2.line(image, (x, y), (x + dx, y + dy), (200, 200, 200), thickness)

    return image

# Add Snow Effect
def add_snow_effect(image, snowflakes=random.randint(100, 150), speed=1):
    height, width, _ = image.shape
    random_size = random.randint(1,2)
    size=(random_size, random_size)
    
    snowflakes_coords = np.zeros((snowflakes, 2), dtype=np.int32)
    snowflakes_coords[:, 0] = np.random.randint(0, width, snowflakes_coords.shape[0])
    snowflakes_coords[:, 1] = np.random.randint(0, height, snowflakes_coords.shape[0])

    for (x, y) in snowflakes_coords:
        snow_color = random.randint(200, 255)  # Random color between gray and white
        cv2.ellipse(image, (x, y), size, 0, 0, 360, (snow_color, snow_color, snow_color), -1)
        y += speed
        if y > height:
            y = 0
            x = np.random.randint(0, width)
        cv2.ellipse(image, (x, y), size, 0, 0, 360, (snow_color, snow_color, snow_color), -1)

    return image


def add_blur(image_HLS, x, y, hw):
    image_copy = np.copy(image_HLS)
    image_copy[y:y+hw, x:x+hw, 1] = image_copy[y:y+hw, x:x+hw, 1] + 1
    image_copy[:, :, 1][image_copy[:, :, 1] > 255] = 255
    image_copy[y:y+hw, x:x+hw, 1] = cv2.blur(image_copy[y:y+hw, x:x+hw, 1], (10, 10))
    return image_copy


def generate_random_blur_coordinates(imshape, hw):
    haze_points = []
    midx = int(imshape[1] / 2)
    midy = int(imshape[0] / 2)    
    index = 1
    while(index < 10):
        x = np.random.randint(midx - 2 * hw, midx + 2 * hw)
        y = np.random.randint(midy - 2 * hw, midy + 2 * hw)
        if x < 0 or y < 0 or x >= imshape[1] - hw or y >= imshape[0] - hw:
            continue
        haze_points.append((x, y))
        index += 1
    return haze_points


# Add Fog Effect
def add_fog_effect(image, foginess=0.5):
    image_HLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    
    # Generate a white image with the same size as the input image
    fog = np.ones_like(image) * 255

    # Calculate the kernel size based on the foginess level
    ksize = max(1, int(foginess * 20))
    if ksize % 2 == 0:
        ksize += 1

    # Apply Gaussian blur to the fog image based on the kernel size
    fog = cv2.GaussianBlur(fog, (ksize, ksize), 0)

    # Blend the fog image with the input image using alpha blending
    alpha = 1 - foginess
    image_with_fog = cv2.addWeighted(image, alpha, fog, foginess, 0)
    
    image_with_fog_HLS = cv2.cvtColor(image_with_fog, cv2.COLOR_RGB2HLS)
    mask = np.zeros_like(image_with_fog)

    imshape = image_with_fog.shape
    hw = 100

    image_with_fog_HLS[:, :, 1] = image_with_fog_HLS[:, :, 1] * 0.8

    haze_list = generate_random_blur_coordinates(imshape, hw)

    for haze_points in haze_list:
        image_with_fog_HLS[:, :, 1][image_with_fog_HLS[:, :, 1] > 255] = 255
        image_with_fog_HLS = add_blur(image_with_fog_HLS, haze_points[0], haze_points[1], hw)

    image_with_fog_RGB = cv2.cvtColor(image_with_fog_HLS, cv2.COLOR_HLS2RGB)
    
    return image_with_fog_RGB


# Example data and parameters
train_directory = "Path/of/train/folder"  # Directory containing training images
batch_size = 32  # Batch size for training
num_images = 5000  # Number of original images to use

# Split train images into training and validation sets
paths = np.array([os.path.join(train_directory, file) for file in os.listdir(train_directory)])
labels = np.array([1 if file.split('.')[0] == 'dog' else 0 for file in os.listdir(train_directory)])

# Randomly select num_images
indices = random.sample(range(len(paths)), num_images)
paths = paths[indices]
labels = labels[indices]

train_paths, val_paths, train_labels, val_labels = train_test_split(paths, labels, test_size=0.2, random_state=42)



num_classes = len(set(labels))  # Number of classes

# Load and preprocess the training images
train_images = []
for path in train_paths:
    image = load_and_preprocess_image(path)
    train_images.append(image)

train_images = np.array(train_images)

# Load and preprocess the validation images
val_images = []
for path in val_paths:
    image = load_and_preprocess_image(path)
    val_images.append(image)

val_images = np.array(val_images)

# Augment the training images with rain or snow effects
augmented_images = []
augmented_labels = []
for ind, image in enumerate(train_images):
    img = image.copy()
    augmentation_type = np.random.choice(['rain', 'snow'])
    if augmentation_type == 'rain':
        augmented_image = add_rain_effect(img)
    else:
        augmented_image = add_snow_effect(img)

    augmented_images.append(augmented_image)
    augmented_labels.append(train_labels[ind])

augmented_images = np.array(augmented_images)
augmented_labels = np.array(augmented_labels)

# Concatenate the original images with augmented images
all_train_images = np.concatenate([train_images, augmented_images], axis=0)
all_train_labels = np.concatenate([train_labels, augmented_labels], axis=0)

# Print the shapes of train_images and train_labels
print(train_images.shape, train_labels.shape)



# Load the InceptionV3 model without the top (fully connected) layers
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
base_model.trainable = False

# Add a custom head on top of the base model
model = models.Sequential()

# Add the InceptionV3 base model
model.add(base_model)

# Add the custom layers
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define the number of training steps and validation steps per epoch
train_steps = len(train_images) // batch_size
val_steps = len(val_images) // batch_size

# Train the model
history = model.fit(
    train_images,
    train_labels,
    batch_size=batch_size,
    steps_per_epoch=train_steps,
    epochs=10,
    validation_data=(val_images, val_labels),
    validation_steps=val_steps
)


model1 = model


from tensorflow.keras.applications import MobileNetV2

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
base_model.trainable = False

# Add a custom head on top of the base model
model = models.Sequential()

# Add the MobileNetV2 base model
model.add(base_model)

# Add the custom layers
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define the number of training steps and validation steps per epoch
train_steps = len(train_images) // batch_size
val_steps = len(val_images) // batch_size

# Train the model
history_MobileNetV2 = model.fit(
    train_images,
    train_labels,
    batch_size=batch_size,
    steps_per_epoch=train_steps,
    epochs=10,
    validation_data=(val_images, val_labels),
    validation_steps=val_steps
)


model2 = model



# Create the Tkinter application
root = tk.Tk()
root.title("Cat and Dog Detection")
root.geometry("400x450")
root.configure(bg='white')
root.resizable(False, False)

# Add a title label
title_label = tk.Label(root, text="Cat and Dog Detection", font=("Arial", 16, "bold"), bg='white')
title_label.pack(pady=10)

# Function to open an image file and make a prediction
def open_image():
    try:
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            image = Image.open(file_path)
            image = image.resize((150, 150))  # Resize the image to match the input size of the model
            photo_image = ImageTk.PhotoImage(image)
            image_label.configure(image=photo_image, highlightthickness=0)
            image_label.image = photo_image

            # Preprocess the image
            image_array = np.array(image)
            image_array = preprocess_input(image_array.astype(np.float32))
            image_array = np.expand_dims(image_array, axis=0)

            # Make a prediction based on the selected model
            if model_selection.get() == 1:
                predictions = model1.predict(image_array)
            else:
                predictions = model2.predict(image_array)

            if predictions[0] > 0.5:
                prediction_text.set("Dog")
                prediction_label.configure(foreground='green')
                prediction_label.pack()  # Show the prediction labels
            else:
                prediction_text.set("Cat")
                prediction_label.configure(foreground='blue')
                prediction_label.pack()  # Show the prediction labels
        else:
            image_label.configure(image='', highlightthickness=0)  # Hide the image and its border
            prediction_label.pack_forget()  # Hide the prediction labels
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Function to handle model selection
def select_model():
    if model_selection.get() == 1:
        print("Inception V3 Model")
    else:
        print("MobileNet V2 Model")

# Create a frame to hold the image label
image_frame = tk.Frame(root, bd=2, relief=tk.SUNKEN)
image_frame.pack(pady=10)

# Create a label to display the image
image_label = tk.Label(image_frame)
image_label.pack()

# Create a variable to store the model selection
model_selection = tk.IntVar(value=1)

# Create a label for model selection
model_label = tk.Label(root, text="Created By Mahdi Savoji", font=("Arial", 8), bg='white')
model_label.pack(side="bottom")


# Create toggle buttons for model selection
model1_button = tk.Radiobutton(root, text="Inception V3 Model", variable=model_selection, value=1, command=select_model, bg='white')
model1_button.pack(side="bottom")
model2_button = tk.Radiobutton(root, text="MobileNet V2 Model", variable=model_selection, value=2, command=select_model, bg='white')
model2_button.pack(side="bottom")

# Create a label for model selection
model_label = tk.Label(root, text="Select Model:", font=("Arial", 12), bg='white')
model_label.pack(side="bottom", pady=10)

# Create a button to open the image file
open_button = tk.Button(root, text="Open Image", command=open_image)
open_button.pack(side="bottom", pady=10)

# Create a label to display the prediction
prediction_label = tk.Label(root, text="Prediction is:", font=("Arial", 14), foreground='white', bg='white')
prediction_label.pack()

# Create a variable to store the prediction text
prediction_text = tk.StringVar()

# Create a label to display the prediction result
prediction_result_label = tk.Label(root, textvariable=prediction_text, font=("Arial", 14), foreground='black', bg='white')
prediction_result_label.pack()

# Run the Tkinter event loop
root.mainloop()
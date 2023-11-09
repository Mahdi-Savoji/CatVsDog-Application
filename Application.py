import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import ImageTk, Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load the trained models
# model1 = tf.keras.models.load_model('my_InceptionV3.h5')
model1 = tf.keras.models.load_model('my_MobileNetV2.h5') # OR ADD Your Model
model2 = tf.keras.models.load_model('my_MobileNetV2.h5')

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
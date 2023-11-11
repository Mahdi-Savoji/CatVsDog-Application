# Cat and Dog Detection Application

<img src="Test/Animation.gif" alt="Cat Vs Dog Application" width="400" height="500">

This is a Tkinter application that uses trained deep learning models to detect whether an image contains a cat or a dog. The user can open an image file, preprocess the image, and make predictions based on the selected model (InceptionV3 or MobileNetV2).

## Requirements

- Python 3.x
- TensorFlow
- Tkinter
- PIL (Python Imaging Library)

You can install the required dependencies using the following command:
pip install tensorflow tkinter pillow


## Usage

1. Clone the repository and navigate to the project directory:
git clone (https://github.com/Mahdi-Savoji/CatVsDog-Application.git)
cd cat-dog-detection

  2. Run the Tkinter application:
     python cat_dog_detection.py

3. The application window will open. You can click the "Open Image" button to select an image file (JPEG, PNG) for prediction.

4. The selected image will be displayed in the application window, and the prediction result (cat or dog) will be shown below.

5. Use the radio buttons to select the desired model (InceptionV3 or MobileNetV2).

## Customization

- You can modify the GUI layout and positioning of the elements according to your preferences by editing the code in `cat_dog_detection.py`.
- To change or add more trained models, you can update the code to load different models and update the prediction logic accordingly.


## Acknowledgments

The code for this application is inspired by various sources and tutorials on Tkinter GUI development and deep learning. Special thanks to the authors and contributors of the following projects:

- TensorFlow: https://www.tensorflow.org/
- Tkinter: https://docs.python.org/3/library/tkinter.html
- PIL (Python Imaging Library): https://pillow.readthedocs.io/

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.

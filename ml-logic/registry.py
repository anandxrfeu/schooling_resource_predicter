import os
from tensorflow.keras.models import load_model


# Load the pre-trained model

def load_model():
    folder_path = "/models/"
    file_name = "model.h5"
    file_path = os.path.join(os.getcwd()+folder_path, file_name)
    return load_model(file_path)

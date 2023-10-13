# Object Detection with TensorFlow

This project is a simple example of object detection using TensorFlow and Keras. It includes a basic Convolutional Neural Network (CNN) model 
for detecting whether "George" is present in an image. The dataset used for training includes both positive and negative samples.

## Getting Started

These instructions will help you get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Python 3.6 or higher
- TensorFlow
- pandas
- numpy
- keras


### Installation

Clone the repository and install the required dependencies:

// bash
git clone https ;

pip install -r requirements.txt

EXPLAIN 
here two datasets
1)grorage.csv
2)non_georage.csv

## Installed necessary Libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image
import os

# Define constants
IMAGE_HEIGHT, IMAGE_WIDTH = 224, 224
NUM_EPOCHS = 10


# Load and prepare data for the further process
georges_df = pd.read_csv('datasets/georges.csv')  
non_georges_df = pd.read_csv('datasets/non-georges.csv') 
# Combine the datasets(merge the two datesets)
dataset = pd.concat([georges_df, non_georges_df], ignore_index=True)

# Split the dataset into training, validation, and test sets
train_df, test_df = train_test_split(dataset, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

# Define data generators for training, validation, and test sets
train_data_generator = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_data_generator = ImageDataGenerator(rescale=1.0/255.0)

test_data_generator = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_data_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='x_col_georges',# Replace with the actual column name for file paths
    y_col='y_col_georges',# Replace with the actual column name for labels
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=32,
    class_mode='binary'
)

val_generator = val_data_generator.flow_from_dataframe(
    dataframe=val_df,
    x_col='x_col_non_georges',
    y_col='y_col_non_georges',
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=32,
    class_mode='binary'
)
test_generator = test_data_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col='x_col_non_georges',
    y_col='y_col_non_georges',
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=32,
    class_mode='binary'
)
# Build and compile the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Train the model
model.fit(train_generator, epochs=NUM_EPOCHS, validation_data=val_generator)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(test_generator)
print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")

# Make predictions on a new image
all_file_paths = georges_df['file_path_column_name'].tolist() + non_georges_df['file_path_column_name'].tolist()
for img_path in all_file_paths:
    img = image.load_img(img_path, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize pixel values
    prediction = model.predict(img_array)
## Condition(check if precision is greater than o.5)
  if prediction > 0.5:
        print("St. George is present in the image.")
    else:
        print("St. George is not present in the image.")

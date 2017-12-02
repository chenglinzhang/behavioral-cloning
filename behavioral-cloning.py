from keras.models import Sequential
from keras.layers import Lambda, Dropout, Flatten, Dense
from keras.layers import Cropping2D, Conv2D

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import numpy as np

import csv
import cv2


# Load dataset with python csv reader
def load_data(data_path):
    lines = []
    with open(data_path) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for line in reader:
            lines.append(line)
    return lines


# Filter extreme steering angles
def filter_data(lines):
    filtered = []
    for line in lines:
        angle = float(line[3])
        if (angle < -0.95 or angle > +0.95):
            continue
        filtered.append(line)
    return filtered


# Read image with OpenCV and convert to RGB format
def read_image(path):
    image_path = './data/' + path.strip()
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   
    return image


# Augment center, left, right of each data line with their flipped images and steering angles
def augment_data(line, correction = 0.20):
    images = []
    angles = []
    
    center_image = read_image(line[0])
    left_image = read_image(line[1])
    right_image = read_image(line[2])
    
    center_angle = float(line[3])
    left_angle = center_angle + correction
    right_angle = center_angle - correction
    
    images.append(center_image)
    angles.append(center_angle)
    images.append(cv2.flip(center_image, 1))
    angles.append(-center_angle)
    
    images.append(left_image)
    angles.append(left_angle)
    images.append(cv2.flip(left_image, 1))
    angles.append(-left_angle)
    
    images.append(right_image)
    angles.append(right_angle)
    images.append(cv2.flip(right_image, 1))
    angles.append(-right_angle)
    
    return images, angles


# Generate data batches with python generator
def generate_data(dataset, batch_size = 32):
    n = len(dataset)
    while True: 
        shuffle(dataset)
        for offset in range(0, n, batch_size):
            batch = dataset[offset : offset + batch_size]

        images = []
        angles = []
        for line in batch:            
            augmented_images, augmented_angles = augment_data(line)
            images.extend(augmented_images)
            angles.extend(augmented_angles)

        x_train = np.array(images)
        y_train = np.array(angles)

        yield x_train, y_train


# Use Nvidia CNN model with dropout
def nvidia_model():
    model = Sequential()
    
    model.add(Cropping2D(cropping = ((60, 20), (0, 0))))
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape = (160, 320, 3)))
    
    model.add(Conv2D(24, 5, strides = (2, 2), activation = 'relu'))
    model.add(Dropout(0.7))
    model.add(Conv2D(36, 5, strides = (2, 2), activation = 'relu'))
    model.add(Conv2D(48, 5, strides = (2, 2), activation = 'relu'))
    model.add(Conv2D(64, 3, activation = 'relu'))
    model.add(Conv2D(64, 3, activation = 'relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(loss = 'mse', optimizer = 'adam')

    return model


# Main to train, validate and save the model
def main():
    model = nvidia_model()
    model.summary()
    
    lines = load_data('./data/driving_log.csv')
    filtered_lines = filter_data(lines)
    
    train_dataset, valid_dataset = train_test_split(filtered_lines, test_size = 0.2)
    train_generator = generate_data(train_dataset, batch_size = 32)
    valid_generator = generate_data(valid_dataset, batch_size = 32)

    model.fit_generator(train_generator,
        steps_per_epoch = len(train_dataset),
        validation_data = valid_generator,
        validation_steps = len(valid_dataset),
        epochs = 21
        )
    
    model.save('model.h5')

if __name__ == "__main__":
    main()


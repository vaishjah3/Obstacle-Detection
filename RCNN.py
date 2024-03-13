import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50

# Sample data generation function
def generate_sample_data(image_size=(100, 100), obstacle_size=(20, 20), num_samples=1000):
    samples = []
    for _ in range(num_samples):
        obstacle_x = np.random.randint(0, image_size[0] - obstacle_size[0])
        obstacle_y = np.random.randint(0, image_size[1] - obstacle_size[1])
        image = np.zeros(image_size)
        image[obstacle_x:obstacle_x+obstacle_size[0], obstacle_y:obstacle_y+obstacle_size[1]] = 1
        samples.append((image, (obstacle_x, obstacle_y, obstacle_x+obstacle_size[0], obstacle_y+obstacle_size[1])))
    return samples

# Generate sample data
image_size = (100, 100)
obstacle_size = (20, 20)
num_samples = 1000
data = generate_sample_data(image_size, obstacle_size, num_samples)

# Split data into train and validation sets
train_data = data[:800]
val_data = data[800:]

# Preprocess data
def preprocess_data(samples):
    X = np.array([sample[0] for sample in samples])
    y = np.array([sample[1] for sample in samples])
    X = X.reshape(-1, image_size[0], image_size[1], 1)  # Reshape for Conv2D
    y = y / np.array([image_size[0], image_size[1], image_size[0], image_size[1]])  # Normalize coordinates
    return X, y

X_train, y_train = preprocess_data(train_data)
X_val, y_val = preprocess_data(val_data)

# Define RCNN model using ResNet50 as backbone
def create_rcnn_model():
    base_model = ResNet50(include_top=False, input_shape=(image_size[0], image_size[1], 3))
    for layer in base_model.layers:
        layer.trainable = False
    roi_input = layers.Input(shape=(4,))
    roi_pooling = layers.ROIPooling(pool_size=(7, 7))([base_model.output, roi_input])
    flatten = layers.Flatten()(roi_pooling)
    fc1 = layers.Dense(512, activation='relu')(flatten)
    fc2 = layers.Dense(128, activation='relu')(fc1)
    output = layers.Dense(4)(fc2)  # 4 for (x1, y1, x2, y2) bounding box coordinates
    model = Model(inputs=[base_model.input, roi_input], outputs=output)
    return model

# Create RCNN model
rcnn_model = create_rcnn_model()

# Compile model
rcnn_model.compile(optimizer=Adam(), loss='mse')

# Train model
rcnn_model.fit([X_train, y_train], y_train, epochs=10, batch_size=32, validation_data=([X_val, y_val], y_val))


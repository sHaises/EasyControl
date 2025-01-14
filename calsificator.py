import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


SEED = 62  # Set seed for reproducibility of the results

# File paths for data, model saving, and TFLite conversion
dataset = 'data.csv'  # Path to the CSV dataset file
model_save_path = 'hand_recognition.keras'  # Path to save the trained model in Keras format
tflite_save_path = 'hand_recognition.tflite'  # Path to save the converted TFLite model

NUM_CLASSES = 3  # Number of gesture classes (e.g., 3 hand gestures)

# Load dataset from CSV
X_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))
Y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))

# Normalize landmarks to a range of 0 to 1
X_dataset = X_dataset / np.max(X_dataset)  # Divide by the max value to normalize

# Split data into training and testing sets (75% for training, 25% for testing)
X_train, X_test, Y_train, Y_test = train_test_split(X_dataset, Y_dataset, train_size=0.75, random_state=SEED)

# Define the architecture of the neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Input((21 * 2,)),  # Input layer: 21 hand landmarks (x, y) coordinates (2 * 21 = 42)
    tf.keras.layers.Dense(64, activation='relu'),  # First dense layer with 64 units and ReLU activation
    tf.keras.layers.Dropout(0.3),  # Dropout layer to prevent overfitting (30% dropout)
    tf.keras.layers.Dense(32, activation='relu'),  # Second dense layer with 32 units and ReLU activation
    tf.keras.layers.Dropout(0.4),  # Dropout layer with 40% dropout
    tf.keras.layers.Dense(16, activation='relu'),  # Third dense layer with 16 units and ReLU activation
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')  # Output layer with softmax activation (multi-class classification)
])

# Callbacks for optimization during training
modelCheckpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    model_save_path,  # Path to save the best model
    verbose=1,
    save_best_only=True,  # Save only the best model based on validation loss
    monitor='val_loss'  # Monitor validation loss for determining best model
)

earlyStopping_callback = tf.keras.callbacks.EarlyStopping(
    patience=20,  # Stop training if validation loss does not improve for 20 epochs
    verbose=1
)

reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',  # Monitor validation loss for adjusting learning rate
    factor=0.5,  # Reduce learning rate by a factor of 0.5
    patience=10,  # Wait for 10 epochs before reducing the learning rate
    verbose=1,
    min_lr=1e-6  # Minimum learning rate to avoid being reduced too much
)

# Compile the model with the Adam optimizer and sparse categorical cross-entropy loss
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,  # Use Adam optimizer
    loss='sparse_categorical_crossentropy',  # Loss function for multi-class classification
    metrics=['accuracy']  # Metrics to evaluate the model performance
)

# Train the model on the training data, with validation on the test data
history = model.fit(
    X_train,  # Training data (features)
    Y_train,  # Training labels
    epochs=1000,  # Maximum number of epochs
    batch_size=128,  # Batch size for training
    validation_data=(X_test, Y_test),  # Validation data (features and labels)
    callbacks=[modelCheckpoint_callback, earlyStopping_callback, reduce_lr_callback]  # List of callbacks
)

# Plot training and validation loss and accuracy over epochs
epochs = history.epoch  # List of epochs
train_loss = history.history['loss']  # Training loss over epochs
val_loss = history.history['val_loss']  # Validation loss over epochs
train_acc = history.history['accuracy']  # Training accuracy over epochs
val_acc = history.history['val_accuracy']  # Validation accuracy over epochs

# Create subplots to display loss and accuracy graphs
plt.figure(figsize=(14, 6))



# Final evaluation of the model on the test data
val_loss, val_acc = model.evaluate(X_test, Y_test, batch_size=128)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_acc}")

# Save the trained model in Keras format
model.save(model_save_path)
print("Model was saved as keras!")  # Output message confirming model saving

# Convert the trained model to TensorFlow Lite format for edge devices
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Apply optimization for better performance on mobile/edge devices
tflite_quantized_model = converter.convert()  # Convert the model to TFLite format

# Save the converted TFLite model
with open(tflite_save_path, 'wb') as f:
    f.write(tflite_quantized_model)

print("TFL save was a sucess!")  # Output message confirming TFLite model saving

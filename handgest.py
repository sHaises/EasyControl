import numpy as np
import tensorflow as tf

class HandGestureRecognition:
    def __init__(self, model_path='hand_recognition.tflite', num_threads=1):
       
        # Initialize the TensorFlow Lite interpreter with the provided model.
        self.interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)
        
        # Allocate memory for input and output tensors.
        self.interpreter.allocate_tensors()

        # Retrieve input and output tensor details.
        self.input_details, self.output_details = self._get_model_details()

    def __call__(self, landmark_list):
       
        # Set the input tensor with the provided landmark list.
        self._set_input_tensor(landmark_list)
        
        # Run inference on the model.
        self.interpreter.invoke()

        # Retrieve the output tensor containing the model's predictions.
        result = self._get_output_tensor()

        # Get the predicted gesture index (the class with the highest probability).
        result_index = self._get_predicted_index(result)
        
        return result_index

    def _get_model_details(self):
      
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        return input_details, output_details

    def _set_input_tensor(self, landmark_list):
  
        # Get the index of the input tensor.
        input_index = self.input_details[0]['index']
        
        # Convert the landmark list to a flattened 1D NumPy array of type float32.
        flat_landmarks = np.array(landmark_list, dtype=np.float32).flatten()
        
        # Reshape the array to match the expected input shape (2D array for a single example).
        input_data = np.expand_dims(flat_landmarks, axis=0)  # Add an extra dimension for batch size (1)
        
        # Set the input tensor with the reshaped data.
        self.interpreter.set_tensor(input_index, input_data)

    def _get_output_tensor(self):
     
        # Get the index of the output tensor.
        output_index = self.output_details[0]['index']
        
        # Retrieve the output tensor containing the model's predictions.
        result = self.interpreter.get_tensor(output_index)
        
        return result

    def _get_predicted_index(self, result):
     
        # Find the index of the maximum value in the output tensor (the predicted class).
        result_index = np.argmax(np.squeeze(result))  # Squeeze to reduce dimensions for easier processing.
        
        return result_index

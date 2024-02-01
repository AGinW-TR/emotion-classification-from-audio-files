"""
This file can be used to try a live prediction. 
"""

import keras
import librosa
import numpy as np

from config import EXAMPLES_PATH
from config import MODEL_DIR_PATH


class LivePredictions:
    """
    Main class of the application.
    """

    def __init__(self, file):
        """
        Init method is used to initialize the main parameters.
        """
        self.file = file
        self.path = MODEL_DIR_PATH + '/Emotion_Voice_Detection_Model.h5'
        self.loaded_model = keras.models.load_model(self.path)

    def make_predictions(self):
        """
        Method to process the files and create your features.
        """
        print(f'Start Predicting {self.file}')

        # Load data with librosa
        data, sampling_rate = librosa.load(self.file, sr=None)

        # Parameters for 3s window with 1s overlap
        window_length = 5 * sampling_rate  # 3 seconds window
        overlap = 2 * sampling_rate        # 1 second overlap
        step_size = window_length - overlap

        # Adjust the loop to handle audio shorter than the window length
        start = 0
        while start < len(data):
            end = min(start + window_length, len(data))
            chunk = data[start:end]
            mfccs = np.mean(librosa.feature.mfcc(y=chunk, sr=sampling_rate, n_mfcc=40).T, axis=0)
            x = np.expand_dims(mfccs, axis=0)
            predictions = self.loaded_model.predict(x, verbose=0)  # No progress bar
            predicted_class = np.argmax(predictions, axis=1)
            emotion = self.convert_class_to_emotion(predicted_class[0])
            print("Prediction for chunk starting at {:.2f} seconds is: {}".format(start / sampling_rate, emotion))

            if end == len(data):  # Break the loop if this is the last chunk
                break

            start += step_size  # Move to the next chunk


    @staticmethod
    def convert_class_to_emotion(pred):
        """
        Method to convert the predictions (int) into human readable strings.
        """
        
        label_conversion = {'0': 'neutral',
                            '1': 'calm',
                            '2': 'happy',
                            '3': 'sad',
                            '4': 'angry',
                            '5': 'fearful',
                            '6': 'disgust',
                            '7': 'surprised'}

        for key, value in label_conversion.items():
            if int(key) == pred:
                label = value
        return label


if __name__ == '__main__':
    live_prediction = LivePredictions(file=EXAMPLES_PATH + '/03-01-01-01-01-02-05.wav')
    live_prediction.make_predictions()

    live_prediction = LivePredictions(file=EXAMPLES_PATH + '/10-16-07-29-82-30-63.wav')
    live_prediction.make_predictions()

    live_prediction = LivePredictions(file=EXAMPLES_PATH + '/trimmed.mp3')
    live_prediction.make_predictions()

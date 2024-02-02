"""
This file can be used to try a live prediction. 
"""

import keras
import librosa
import os 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


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

    def make_predictions(self, sample_len=2, overlap=1):
        print(f'Start Predicting {self.file}')

        # Load data with librosa
        data, sampling_rate = librosa.load(self.file, sr=None)

        # Collecting data for plots
        energy_map = {'neutral': 3, 'calm': 1, 'happy': 7, 'sad': 2, 'angry': 9, 'fearful': 8, 'disgust': 5, 'surprised': 8}
        time_stamps = []
        energy_scores = []

        # Parameters for 3s window with 1s overlap
        window_length = int(sample_len * sampling_rate)  # 3 seconds window
        overlap = int(overlap * sampling_rate)        # 1 second overlap
        step_size = window_length - overlap

        start = 0
        while start < len(data):
            end = min(start + window_length, len(data))
            chunk = data[start:end]
            mfccs = np.mean(librosa.feature.mfcc(y=chunk, sr=sampling_rate, n_mfcc=40).T, axis=0)
            x = np.expand_dims(mfccs, axis=0)
            predictions = self.loaded_model.predict(x, verbose=0)
            predicted_class = np.argmax(predictions, axis=1)
            emotion = self.convert_class_to_emotion(predicted_class[0])

            # Collecting data for energy plot
            time_stamps.append(start / sampling_rate)
            energy_scores.append(energy_map[emotion])

            print("Prediction for chunk starting at {:.2f} seconds is: {}".format(start / sampling_rate, emotion))

            if end == len(data):
                break

            start += step_size

        # Plotting
        self.plot_data(data, sampling_rate, time_stamps, energy_scores)

    def plot_data(self, audio_data, sampling_rate, time_stamps, energy_scores):
        fig = plt.figure(figsize=(10, 5))

        # Create a GridSpec with 3 rows and 1 column with custom height ratios
        gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1])

        # Mel Spectrogram
        ax0 = plt.subplot(gs[0])
        S = librosa.feature.melspectrogram(y=audio_data, sr=sampling_rate)
        img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max), sr=sampling_rate, y_axis='mel', ax=ax0)
        ax0.set_xticklabels([])

        # 1D Sound Wave
        ax1 = plt.subplot(gs[1])
        ax1.plot(np.linspace(0, len(audio_data) / sampling_rate, len(audio_data)), audio_data)
        ax1.set_ylabel('Amplitude')
        ax1.set_xticklabels([])

        # Emotion Energy Over Time
        ax2 = plt.subplot(gs[2])
        ax2.plot(time_stamps, energy_scores, marker='o')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Energy Level')

        # Adjust layout and display the combined plot            
        plt.subplots_adjust(wspace=0, hspace=0)
        # plt.show()


        save_name = os.path.basename(live_prediction.file).split('.')[0] + '.jpg'
        plt.savefig(f'figs/{save_name}')



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

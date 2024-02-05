import os
import time
import joblib
import librosa
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from config import SAVE_DIR_PATH
from config import TRAINING_FILES_PATH


class CreateFeatures:
    def __init__(self):
        if not os.path.exists('static'): os.mkdir('static')
        if not os.path.exists('joblib_features'): os.mkdir('joblib_features')
        if not os.path.exists('uploads'): os.mkdir('uploads')
        if not os.path.exists('figs'): os.mkdir('figs')

    @staticmethod
    def features_creator(path, save_dir) -> str:
        lst = []
        # scaler = StandardScaler()  # For normalization

        start_time = time.time()

        # Count total number of files
        total_files = sum([len(files) for r, d, files in os.walk(path)])

        # Start processing with tqdm
        pbar = tqdm(total=total_files, desc="Processing files")
        
        for subdir, dirs, files in os.walk(path):
            for file in files:
                try:
                    X, sample_rate = librosa.load(os.path.join(subdir, file), res_type='kaiser_fast')
                    mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T
                    # mfccs_scaled = scaler.fit_transform(mfccs.T)  # Normalizing the MFCCs
                    mfccs_mean = np.mean(mfccs, axis=0)

                    file = int(file[7:8]) - 1
                    arr = mfccs_mean, file
                    lst.append(arr)
                except ValueError as err:
                    print(f"Error processing {file}: {err}")
                    continue
                    
                finally:
                    pbar.update(1)  # Update progress bar                    

        print("--- Data loaded. Loading time: %s seconds ---" % (time.time() - start_time))

        X, y = zip(*lst)
        X, y = np.asarray(X), np.asarray(y)

        print(X.shape, y.shape)

        X_name, y_name = 'X.joblib', 'y.joblib'
        joblib.dump(X, os.path.join(save_dir, X_name))
        joblib.dump(y, os.path.join(save_dir, y_name))

        return "Completed"


if __name__ == '__main__':
    print('Routine started')
    FEATURES = CreateFeatures.features_creator(path=TRAINING_FILES_PATH, save_dir=SAVE_DIR_PATH)
    print('Routine completed.')

import librosa
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

# CHANGE FILEPATH AS NEEDED!
df = pd.read_csv('E:/Downloads/train_extended.csv') # absolute file path to train_extended.csv
df = df[['rating', 'ebird_code', 'duration', 'filename', 'species']] # keep only useful data

# clean some of the data using the rating of audio
df = df[df['rating'] >= 2.5]
df = df.reset_index()

# map species name to a number for labelling purposes
speciesToLabel = {}
label = 0
for ind in df.index:
    if df['species'][ind] in speciesToLabel:
        df.at[ind, 'label'] = speciesToLabel[df['species'][ind]]
    else:
        speciesToLabel[df['species'][ind]] = label
        df.at[ind, 'label'] = label
        label = label + 1

# number of species to classify, comment out these 3 lines if you want to classify all
df = df[df['label'] < 4]
df = df.reset_index()
df.to_csv('audio_test.csv')

# shuffle the dataframe
df = df.sample(frac=1).reset_index(drop=True)
size = len(df['label'])
df_train = df.iloc[:int(size * 0.7)] # select first 70% of data as training data
#df_train = df.iloc[:500]
df_test = df.iloc[int(-0.3 * size):] # select last 30% of data as testing data
#df_test = df.iloc[-25:]


# get filepaths and create parallel array for labels
train_filenames = []
train_labels = []
for ind in df_train.index:
    path = df_train['ebird_code'][ind] + '/' + df_train['filename'][ind]
    train_filenames.append(path)
    train_labels.append(int(df_train['label'][ind]))

test_filenames = []
test_labels = []
for ind in df_test.index:
    path = df_test['ebird_code'][ind] + '/' + df_test['filename'][ind]
    test_filenames.append(path)
    test_labels.append(int(df_test['label'][ind]))

# extract features, current does mfcc and chroma
def extract_features(filepath):
    signal, sample_rate = librosa.load(filepath, offset=0.5, duration=5) # load 5 seconds of audio from 0.5 seconds from beginning
    df_mfccs = pd.DataFrame()  # create a dataframe for the mfcc features
    mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=12)  # extract 12 mfcc features
    # iterate through the 12 mfcc features and create a columns in the dataframe to store the vectors
    for n_mfcc in range(len(mfccs)):
        df_mfccs['MFCC_%d' % (n_mfcc + 1)] = mfccs[n_mfcc]

    df_chroma = pd.DataFrame()  # create a dataframe for the chroma features
    chromagram = librosa.feature.chroma_stft(y=signal,
                                             sr=sample_rate)  # user chroma_stft function to extract chroma features
    # iterate through resulting matrix and store each vector in its own column in the dataframe
    for n_chroma in range(len(chromagram)):
        df_chroma['Chroma_%d' % (n_chroma + 1)] = chromagram[n_chroma]

    feature_matrix = pd.concat([df_mfccs, df_chroma], axis=1) # concatenation

    # average every 10 time windows
    df_1 = feature_matrix.iloc[1:10, :]
    df_2 = feature_matrix.iloc[11:20, :]
    df_3 = feature_matrix.iloc[21:30, :]
    df_4 = feature_matrix.iloc[31:40, :]
    df_1 = df_1.reset_index()
    df_2 = df_2.reset_index()
    df_3 = df_3.reset_index()
    df_4 = df_4.reset_index()
    feature_matrix = pd.concat([df_1, df_2, df_3, df_4], axis=1)
    feature_matrix = feature_matrix.mean(axis=0)
    feature_matrix = feature_matrix.reset_index()

    # create feature vector
    feature_vector = [];
    for i in feature_matrix[0]:
        feature_vector.append(i);

    return feature_vector


x1 = []
y1 = []
x2 = []
y2 = []

# CHANGE ACCORDINGLY!
prefix = 'E:/Downloads/audio/' # file path to audio folder, inside the audio folder are folders with of each species


# function to append to training data by inputting audio file to extract features from and its label
def create_training_set(filenames, y_labels):
    for i in range(len(filenames)):
        if i % 10 == 0:
            print('Finished ' + str(i))
        try:
            x1.append(extract_features(prefix + filenames[i]))
            y1.append(y_labels[i])
        except:
            print(filenames[i] + " failed")


def create_testing_set(filenames, y_labels):
    for i in range(len(filenames)):
        if i % 10 == 0:
            print('Finished ' + str(i))
        try:
            x2.append(extract_features(prefix + filenames[i]))
            y2.append(y_labels[i])
        except:
            print(filenames[i] + " failed")


# create training and test sets
create_training_set(train_filenames, train_labels)
create_testing_set(test_filenames, test_labels)

# convert training data to numpy array
x_train = np.array(x1)
y_train = np.array(y1)

# convert test data to numpy array
x_test = np.array(x2)
y_test = np.array(y2)

# train model
nbc = GaussianNB()
nbc.fit(x_train, y_train)
y_nbc_predicted = nbc.predict(x_test)

#print classification report
print("\nNaive Bayes")
print(classification_report(y_test, y_nbc_predicted, zero_division=0))

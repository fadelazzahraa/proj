
import numpy as np
import pandas as pd 
import librosa
import librosa.display
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("begin")
model_new = tf.keras.models.load_model("res_model.h5")
model_new.summary()
print("begin 2")

def zcr(data,frame_length=2048,hop_length=512):
    zcr=librosa.feature.zero_crossing_rate(y=data,frame_length=frame_length,hop_length=hop_length)
    return np.squeeze(zcr)
def rmse(data,frame_length=2048,hop_length=512):
    rmse=librosa.feature.rms(y=data,frame_length=frame_length,hop_length=hop_length)
    return np.squeeze(rmse)
def mfcc(data,sr,frame_length=2048,hop_length=512,flatten:bool=True):
    mfcc=librosa.feature.mfcc(y=data,sr=sr)
    return np.squeeze(mfcc.T)if not flatten else np.ravel(mfcc.T)

def extract_features(data,sr,frame_length=2048,hop_length=512):
    result=np.array([])

    result=np.hstack((result,
                      zcr(data,frame_length,hop_length),
                      rmse(data,frame_length,hop_length),
                      mfcc(data,sr,frame_length,hop_length)
                     ))
    return result

path="scream_aaa-1-92047.mp3"
data,sr=librosa.load(path,duration=2.5,offset=0.6)

aud=extract_features(data,sr)
aud=[aud]

dfpredict=pd.DataFrame(aud)

dfpredict=dfpredict.fillna(0)

scaler=StandardScaler()
predict_data=scaler.fit_transform(dfpredict)

predict_data=np.expand_dims(predict_data,axis=2)

data_pred = model_new.predict(predict_data)

data_pred = np.argmax(data_pred, axis=1)
print(data_pred)
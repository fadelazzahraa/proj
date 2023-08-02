from PIL import Image, ImageTk
import tkinter as tk
import numpy as np
import pandas as pd
import librosa
import librosa.display
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

import sounddevice as sound
from scipy.io.wavfile import write
import time

import warnings
warnings.filterwarnings('ignore')

print("App begin")
model_new = tf.keras.models.load_model("res_model.h5")
model_new.summary()
print("Model loaded")


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

def resize_image(image_path, width, height):
    # Mengubah ukuran gambar sesuai lebar (width) dan tinggi (height) yang diinginkan
    image = Image.open(image_path)
    resized_image = image.resize((width, height), Image.ANTIALIAS)
    return ImageTk.PhotoImage(resized_image)


def start_recording():
    print("Record started")

    freq=44100
    try:
        dur=int(duration.get())
    except:
        dur=5
    recording=sound.rec(dur*freq,samplerate=freq,channels=2)

    temp=dur

    status_label.config(text="Recording...")
    while temp>0:
        root.update()
        time.sleep(1)
        temp -= 1

        if (temp==0):
            status_label.config(text="Record Selesai")

        status_label.config(text=f"{temp}")

    status_label.config(text="Record Selesai")
    sound.wait()
    write("record.wav", freq,recording)


def stop_recording():
    status_label.config(text="Record Selesai")

def analyze_emotion():
    # Fungsi ini akan dijalankan ketika tombol "Ambil Hasil Emosi" ditekan
    # Di sini Anda dapat menambahkan kode untuk menganalisis hasil emosi
    # dan menampilkan hasilnya pada label "result_label" dan "suggestion_label"
    path="record.wav"
    data,sr=librosa.load(path,duration=2.5,offset=0.6)

    aud=extract_features(data,sr)
    aud=[aud]

    dfpredict=pd.DataFrame(aud)

    dfpredict=dfpredict.fillna(0)

    scaler=StandardScaler()
    predict_data=scaler.fit_transform(dfpredict)

    predict_data=np.expand_dims(predict_data,axis=2)

    data_pred = model_new.predict(predict_data)

    class_data_pred = np.argmax(data_pred, axis=1)
    idx_data_pred = class_data_pred[0]
    acc_data_pred = data_pred[0][class_data_pred[0]]

    # Contoh hasil emosi dan akurasinya
    list_result_emotion = ['Marah', 'Jijik', 'Takut', 'Bahagia', 'Netral', 'Sedih', 'Kaget']
    list_suggestion = [
        'Jangan marah marah', 
        'Jijik hanya ada di pikiran Anda saja, jangan jijik!', 
        'Tenang, Anda tidak sendiri!', 
        'Jangan cuma bahagia sendirian ya!', 
        'Tetap tenang dan netral', 
        'Jangan bersedih, semangat dong!', 
        'Kagetnya jangan berlebihan ya!']

    result_label.config(text=f"Hasil Emosi: {list_result_emotion[idx_data_pred]} (Akurasi: {'{:.2f}'.format( acc_data_pred )})")
    suggestion_label.config(text=f"Saran: {list_suggestion[idx_data_pred]}")




# Membuat jendela utama
root = tk.Tk()
root.title("YourEmo")

# Mengatur ukuran jendela (lebar x tinggi)
root.geometry("800x600")

# Membuat label untuk menampilkan gambar
# Ganti "nama_gambar.png" dengan nama berkas gambar Anda dan pastikan berkas gambar ada dalam direktori yang sama dengan skrip Python Anda.
image_width = 200
image_height = 170
image_path = "microphone.jpg"
photo = resize_image(image_path, image_width, image_height)
image_label = tk.Label(root, image=photo)
image_label.pack(pady=10)

# Membuat label untuk menampilkan status rekaman
status_label = tk.Label(root, text="Tidak merekam", font=("Arial", 14))
status_label.pack()

duration=tk.StringVar()
entry=tk.Entry(root,textvariable=duration, font=("Arial", 12), width=15).pack(pady=10)
tk.Label(text="Masukkan detik rekaman", font=("Arial", 12), background="#4a4a4a", fg="white").pack()

# Membuat tombol "Record" dan "Ambil Hasil Emosi"
record_button = tk.Button(root, text="Rekam!", command=start_recording, font=("Arial", 12))
record_button.pack(pady=5)


analyze_button = tk.Button(root, text="Analisa Hasil Emosi Rekaman!", command=analyze_emotion, font=("Arial", 12))
analyze_button.pack(pady=5)

# Membuat label untuk menampilkan hasil emosi dan saran atas hasil emosi
result_label = tk.Label(root, text="", font=("Arial", 12))
result_label.pack(pady=10)

suggestion_label = tk.Label(root, text="", font=("Arial", 12))
suggestion_label.pack(pady=5)

# Menjalankan program
print("Form loaded")
root.mainloop()
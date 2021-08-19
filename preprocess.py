import glob

import numpy as np
import pandas as pd
import cv2

import soundfile

from python_speech_features import fbank
from scipy.signal import spectrogram
from os import path, mkdir
import os
from tqdm import tqdm
import math

from pydub import AudioSegment
from pydub.utils import make_chunks
AudioSegment.converter = str(path.join("C:/FFmpeg", "bin/ffmpeg.exe"))

def split_file(src, dst, new_len_ms):
    if not path.isdir(dst):
        mkdir(dst)
    print(f"Extracting from {src} to {dst}")
    ending = src.split('.')[-1]
    my_audio = AudioSegment.from_file(src, ending)
    chunks = make_chunks(my_audio, new_len_ms)

    for i, chunk in enumerate(chunks):
        file_name = src.split('\\')[-1]
        chunk_name = f"{dst}\\{file_name.split('.')[0]}_frame{i}.{ending}"
        print(f"Extracting {chunk_name}")
        chunk.export(chunk_name, format=ending)

def rescale_for_model(fbanks):
    pass


def make_fbank(wav, fs=22050):
    winlen = 1. / 43.0664 # specRes_Hz from model 
    winstep = 2.9 / 1000. # tempRes_ms from model
    nfft = 1024
    preemph = 0.5
    M, _ = fbank(wav, samplerate=fs,
                 nfilt=41, nfft=nfft,
                 lowfreq=0, highfreq=11025,
                 preemph=0.5,
                 winlen=winlen, winstep=winstep,
                 winfunc=lambda x: np.hanning(x))

    logM = np.log(M)
    logM = np.swapaxes(logM, 0, 1)

    targetSize = 682
    cut = np.minimum(logM.shape[1], targetSize)
    background = np.float64(logM[:,:cut]).mean(axis=1)

    features = np.float32(np.float64(logM) - background[:, np.newaxis])

    if features.shape[1] < targetSize:
        features = np.concatenate((features,
                                   np.zeros((features.shape[0],
                                             targetSize-features.shape[1]),
                                            dtype='float32')), axis=1)
    elif features.shape[1] > targetSize:
        features = features[:,:(targetSize-features.shape[1])]

    return features

def __process_csv(df):
    df["start.time"] = df["start.time"].apply(lambda x: x*10)
    df["end.time"] = df["end.time"].apply(lambda x: x*10)
    return df

def __ceil(x, s):
    return s * math.ceil(float(x)/s)

def __floor(x, s):
    return s * math.floor(float(x)/s)

def preprocess_data(target="resources/ANTBOK_training_labels.csv", audio_dir = "resources/training/clips", DEBUG=False):
    assert path.exists(target)
    f = open(target)
    window_len = 2 # seconds
    step = 0.1 # window scroll speed

    # Remove any positive label from a window where the call is shorter than this.
    minimum_call_seconds = 0.1 # seconds

    target = np.array([], dtype=np.int8)
    data = None

    df = pd.read_csv(f)
    f.close()
    if DEBUG:
        df = __process_csv(df)
    df["duration"] = df["end.time"] - df["start.time"]
    csv = df[df["duration"] > minimum_call_seconds]

    name = "filename"
    if name not in df.columns:
        name = "ImageName"

    vfloor = np.vectorize(__floor)
    vceil = np.vectorize(__ceil)

    for f in tqdm(os.listdir(audio_dir)):
        f = f"{audio_dir}/{f}"
        s = path.splitext(path.basename(f))[0] + '.png'
        boxes = csv[csv[name] == s]
        boxes = boxes[boxes["annotation"] == "Bachmans Sparrow"]
        # read the audio file. if this is a stereo file, 
        # we take only the first channel.
        wav, fs = soundfile.read(f)
        if np.ndim(wav) > 1:
            wav = wav[:,0]
        samples_per_window = fs * window_len
        # generate all 2 seconds windows as a list, then round down
        # the label start time to the nearest 10 second window.
        # all_windows = np.arange(
        #     0, np.ceil(len(wav) / samples_per_window).astype(np.int), step)
        all_windows = np.arange(0, len(wav), step*fs).astype(np.int)

        if len(boxes["start.time"]) > 0:
            positive_start_windows = ((vfloor(boxes["start.time"]*(step*fs), step*fs))).astype(np.int).tolist()
            positive_end_windows = ((vceil(boxes["end.time"]*(step*fs), step*fs))).astype(np.int).tolist()
            positive_windows = []
            for i in range(len(positive_start_windows)):
                start = positive_start_windows[i]
                end = positive_end_windows[i]
                end_index = len(all_windows)
                if end < len(wav):
                    try:
                        end_index = all_windows.tolist().index(end)
                    except:
                        print(all_windows)
                        raise
                frame = start
                while frame <= end:
                    positive_windows.append(frame)
                    frame += step*fs
            positive_windows = np.array(positive_windows)
        else:
            positive_windows = np.array([])
        try:
            positive_indices = [all_windows.tolist().index(pw) if pw < len(wav) else len(all_windows)-1 for pw in positive_windows]
        except:
            print(f"all_windows:\n{all_windows}\n")
            print(f"positive_windows:\n{positive_windows}\n")
            raise

        targets = [0 for w in all_windows]
        for pi in positive_indices:
            targets[pi] = 1
        for t in targets:
            target = np.append(target, t)
        for i,w in enumerate(all_windows):
            start = i
            end = start + samples_per_window
                     
            window = wav[start:end]

            if len(window) == 0:
                print(f"DEBUGGING: {start}:{wav[start]} ==> {end}:{wav[end]}")
            
            if data is None:
                data = np.array([[window, f, w]])
            else:
                data = np.append(data, [[window, f, w]], axis=0)
    return data, target

def process_new_data(src):
    window_len = 2

    wav, fs = soundfile.read(src)
    if np.ndim(wav) > 1:
        wav = wav[:,0]
    samples_per_window = fs*window_len

    all_windows = np.arange(0, np.ceil(len(wav) / samples_per_window).astype(np.int))

    data = None

    for w in all_windows:
        start = w * samples_per_window
        end = start + samples_per_window

        window = wav[start:end]

        if data is None:
            data = np.array([[window, src, w]])
        else:
            data = np.append(data, [[window, src, w]], axis=0)
    return data

def sliding_windows(image, im_dim, step, ws):
    """
    Generate and yield sliding window images from original image.

    Parameters:
    image - 2D Integer Numpy array of original image
    im_dim - 2D Integer Tuple of width, height for the original image
    step - Integer step size, how many pixels are "skipped" between each window
    ws - 2D Integer Tuple of width, height for the window size
    """

    for y in range(0, im_dim[0] - ws[0], step):
        for x in range(0, im_dim[1] - ws[1], step):
            yield (x, y, image[y:y + ws[0], x:x + ws[1]])

def image_pyramid(image, im_dim, scale=1.5, minsize=(224,224)):
    yield image

    while True:
        w = int(im_dim[1] / scale)
        image = cv2.resize(image, (w, im_dim[0]))

        if im_dim[0] < minsize[1] or im_dim[1] < minsize[0]:
            break

        yield image

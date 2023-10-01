import tensorflow as tf
try: [tf.config.experimental.set_memory_growth(gpu, True) for gpu in tf.config.experimental.list_physical_devices("GPU")]
except: pass
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# tf.keras.mixed_precision.set_global_policy('mixed_float16')

import os
import tarfile
import pandas as pd
from tqdm import tqdm
from urllib.request import urlopen
from io import BytesIO

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from mltu.preprocessors import WavReader

from mltu.tensorflow.dataProvider import DataProvider
from mltu.transformers import LabelIndexer, LabelPadding, SpectrogramPadding
from mltu.tensorflow.losses import CTCloss
from mltu.tensorflow.callbacks import Model2onnx, TrainLogger
from mltu.tensorflow.metrics import CERMetric, WERMetric

from model import train_model
from configs import ModelConfigs


def downloadExtract(url, extractTo="dataset", chunkSize=1024*1024):
    response = urlopen(url)

    data = b""
    iterations = response.length // chunkSize + 1
    for _ in tqdm(range(iterations)):
        data += response.read(chunkSize)

    tarFile = tarfile.open(fileobj=BytesIO(data), mode="r|bz2")
    tarFile.extractall(path=extractTo)
    tarFile.close()


# datasetPath = os.path.join("dataset", "LJSpeech-1.1")
# if not os.path.exists(datasetPath):
#     downloadExtract('https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2')

datasetPath = "datasets"
pathEN = datasetPath + "/metadata_EN.csv"
pathVI = datasetPath + "/metadata_VI.csv"
wavsPath = datasetPath + "/wavs/"

# Read metadata file and parse it
metadataEN = pd.read_csv(pathEN, sep="|", header=None, quoting=3)
metadataEN.columns = ["file_name", "transcription", "normalized_transcription"]
metadataEN = metadataEN[["file_name", "normalized_transcription"]]

metadataVI = pd.read_csv(pathVI, sep="|", header=None, quoting=3)
metadataVI.columns = ["file_name", "transcription", "normalized_transcription"]
metadataVI = metadataVI[["file_name", "normalized_transcription"]]

# Structure the dataset where each row is a list of [wav file path, sound transcription]
datasetEN = [[f"datasets/wavs/{file}.wav", label.lower()] for file, label in metadataEN.values.tolist()]
datasetVI = [[f"datasets/wavs/{file}.wav", label.lower()] for file, label in metadataVI.values.tolist()]

dataset = datasetEN + datasetVI

# Create a ModelConfigs object to store model configurations
configs = ModelConfigs()

max_text_length, max_spectrogram_length = 0, 0
for filePath, label in tqdm(dataset):
    spectrogram = WavReader.get_spectrogram(
        filePath, 
        frame_length=configs.frame_length, 
        frame_step=configs.frame_step,
        fft_length=configs.fft_length
    )
    validLabel = [c for c in label if c in configs.vocab]
    max_text_length = max(max_text_length, len(validLabel))
    max_spectrogram_length = max(max_spectrogram_length, spectrogram.shape[0])
    configs.input_shape = [max_spectrogram_length, spectrogram.shape[1]]

configs.max_spectrogram_length = max_spectrogram_length
configs.max_text_length = max_text_length
configs.save()

# Create a data provider for the dataset
data_provider = DataProvider(
    dataset,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[
        WavReader(frame_length=configs.frame_length, frame_step=configs.frame_step, fft_length=configs.fft_length),
    ],
    transformers=[
        SpectrogramPadding(max_spectrogram_length=max_spectrogram_length, padding_value=0),
        LabelIndexer(configs.vocab),
        LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab))
    ]
)

# Split the dataset into training and validation sets
train_data_provider, val_data_provider = data_provider.split(split = 0.9)

# Creating TensorFlow model architecture
model = train_model(
    input_dim = configs.input_shape,
    output_dim = len(configs.vocab),
    dropout=0.5
)

# Compile the model and print summary
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=configs.learning_rate), 
    loss=CTCloss(), 
    metrics=[
        CERMetric(vocabulary=configs.vocab),
        WERMetric(vocabulary=configs.vocab)
        ],
    run_eagerly=False
)
model.summary(line_length=110)

# Define callbacks
earlystopper = EarlyStopping(monitor="val_CER", patience=20, verbose=1, mode="min")
checkpoint = ModelCheckpoint(f"{configs.model_path}/model.h5", monitor="val_CER", verbose=1, save_best_only=True, mode="min")
trainLogger = TrainLogger(configs.model_path)
tb_callback = TensorBoard(f"{configs.model_path}/logs", update_freq=1)
reduceLROnPlat = ReduceLROnPlateau(monitor="val_CER", factor=0.8, min_delta=1e-10, patience=5, verbose=1, mode="auto")
model2onnx = Model2onnx(f"{configs.model_path}/model.h5")

# Train the model
model.fit(
    train_data_provider,
    validation_data=val_data_provider,
    epochs=configs.train_epochs,
    callbacks=[earlystopper, checkpoint, trainLogger, reduceLROnPlat, tb_callback, model2onnx],
    workers=configs.train_workers
)

# Save training and validation datasets as csv files
train_data_provider.to_csv(os.path.join(configs.model_path, "train.csv"))
val_data_provider.to_csv(os.path.join(configs.model_path, "val.csv"))
# Databricks notebook source
# MAGIC %md
# MAGIC # Installing Whisper
# MAGIC 
# MAGIC The commands below will install the Python packages needed to use Whisper models and evaluate the transcription results.

# COMMAND ----------

! pip install git+https://github.com/openai/whisper.git
! pip install jiwer

# COMMAND ----------

# MAGIC %md
# MAGIC # Loading the LibriSpeech dataset
# MAGIC 
# MAGIC The following will load the test-clean split of the LibriSpeech corpus using torchaudio.

# COMMAND ----------

import os
import numpy as np

try:
    import tensorflow  # required in Colab to avoid protobuf compatibility issues
except ImportError:
    pass

import torch
import pandas as pd
import whisper
import torchaudio

from tqdm.notebook import tqdm


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# COMMAND ----------

class LibriSpeech(torch.utils.data.Dataset):
    """
    A simple class to wrap LibriSpeech and trim/pad the audio to 30 seconds.
    It will drop the last few seconds of a very small portion of the utterances.
    """
    def __init__(self, split="test-clean", device=DEVICE):
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=os.path.expanduser("~/.cache"),
            url=split,
            download=True,
        )
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        audio, sample_rate, text, _, _, _ = self.dataset[item]
        assert sample_rate == 16000
        audio = whisper.pad_or_trim(audio.flatten()).to(self.device)
        mel = whisper.log_mel_spectrogram(audio)
        
        return (mel, text)

# COMMAND ----------

dataset = LibriSpeech("test-clean")
loader = torch.utils.data.DataLoader(dataset, batch_size=16)

# COMMAND ----------

# MAGIC %md
# MAGIC # Running inference on the dataset using a base Whisper model
# MAGIC 
# MAGIC The following will take a few minutes to transcribe all utterances in the dataset.

# COMMAND ----------

model = whisper.load_model("base.en")
print(
    f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
    f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
)

# COMMAND ----------

# predict without timestamps for short-form transcription
options = whisper.DecodingOptions(language="en", without_timestamps=True)

# COMMAND ----------

hypotheses = []
references = []

for mels, texts in tqdm(loader):
    results = model.decode(mels, options)
    hypotheses.extend([result.text for result in results])
    references.extend(texts)

# COMMAND ----------

data = pd.DataFrame(dict(hypothesis=hypotheses, reference=references))
data

# COMMAND ----------

# MAGIC %md
# MAGIC # Calculating the word error rate
# MAGIC 
# MAGIC Now, we use our English normalizer implementation to standardize the transcription and calculate the WER.

# COMMAND ----------

import jiwer
from whisper.normalizers import EnglishTextNormalizer

normalizer = EnglishTextNormalizer()

# COMMAND ----------

data["hypothesis_clean"] = [normalizer(text) for text in data["hypothesis"]]
data["reference_clean"] = [normalizer(text) for text in data["reference"]]
data

# COMMAND ----------

wer = jiwer.wer(list(data["reference_clean"]), list(data["hypothesis_clean"]))

print(f"WER: {wer * 100:.2f} %")

import re
import time
import whisper
from whisper.tokenizer import get_tokenizer
from whisper.timing import add_word_timestamps

def extract_number(file_name):
    match = re.search(r'(\d+)_', file_name)
    if match:
        return int(match.group(1))
    else:
        return None

def download_model(setting = "medium"):
    print("Downloading the model if needed. ")
    model = whisper.load_model(setting)
    return model

def transcribe_test(model,tokenizer,input_audio, languageOp, prompt=""):
    SAMPLE_RATE = 16000
    print("transcribe_test for "+ input_audio)
    prepend_punctuations: str = "\"'“¿([{-",
    append_punctuations: str = "\"'.。,，!！?？:：”)]}、",

    audio = whisper.load_audio(input_audio)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    options = whisper.DecodingOptions(prompt=prompt, max_initial_timestamp=None, without_timestamps=False, fp16= False)
    result = whisper.decode(model, mel, options)

    #global tokenizer
    if tokenizer is None:
      tokenizer = get_tokenizer(multilingual=model.is_multilingual, language=languageOp, task=options.task)

    text_tokens = [tokenizer.decode([t]) for t in result.tokens]

    starttime=time.time()
    segments = [{"seek": 0, "start": 0, "end": len(audio) / SAMPLE_RATE, "tokens": result.tokens}]
    add_word_timestamps(
        segments=segments,
        model=model,
        tokenizer=tokenizer,
        mel=mel,
        num_frames=mel.shape[-1],
        prepend_punctuations=prepend_punctuations,
        append_punctuations=append_punctuations,
        last_speech_timestamp=0.0,
    )
    word_timestamps = segments[0]["words"]

    print(f"time: {time.time() - starttime}")

    return result.text, text_tokens, word_timestamps

from utilities.file_utils import get_file_list
tokenizer = None
model = download_model()
list_files = get_file_list()
transcribe_test(model,tokenizer,)


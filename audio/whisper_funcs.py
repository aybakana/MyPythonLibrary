import json, os, subprocess, re, whisper, time, csv
from whisper.tokenizer import get_tokenizer
from whisper.timing import add_word_timestamps
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.editor import concatenate_videoclips, concatenate_audioclips
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

SAMPLE_RATE = 16000
tokenizer = None

def download_model(setting = "medium"):
    print("Downloading the model if needed. ")
    model = whisper.load_model(setting)
    return model


def extract_number(file_name):
    match = re.search(r'(\d+)_', file_name)
    if match:
        return int(match.group(1))
    else:
        return None
def transcribe_test(model,input_audio, languageOp, prompt=""):
    print("transcribe_test for "+ input_audio)
    prepend_punctuations: str = "\"'“¿([{-",
    append_punctuations: str = "\"'.。,，!！?？:：”)]}、",

    audio = whisper.load_audio(input_audio)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    options = whisper.DecodingOptions(prompt=prompt, max_initial_timestamp=None, without_timestamps=False, fp16= False)
    result = whisper.decode(model, mel, options)

    global tokenizer
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

def extract_sentences(sentence_timestamps,offset,ignoreFirst=False,ignoreLast=False):
    # Extracting sentences from words list
    sentences = dict()
    current_sentence = []
    startingIndex = 0
    ignore_current = False
    punctuation_marks = ['.', '!', '?']
    for index ,wordSection in enumerate(sentence_timestamps):
        if wordSection["word"][-1:] == "." or wordSection["word"][-1:] == "!" or wordSection["word"][-1:] == "?":
            current_sentence.append(wordSection["word"])
            if current_sentence:
              if index == len(sentence_timestamps)-1 and not ignoreLast:  # Last one
                sentences[" ".join(current_sentence)] = [offset+sentence_timestamps[startingIndex]["start"], offset+wordSection["end"]]
              elif index == 0 and not ignoreFirst:  # First one
                sentences[" ".join(current_sentence)] = [offset+sentence_timestamps[startingIndex]["start"], offset+wordSection["end"]]
              else:
                sentences[" ".join(current_sentence)] = [offset+sentence_timestamps[startingIndex]["start"], offset+wordSection["end"]]
            current_sentence = []
        else:
            if current_sentence == []:
                startingIndex = index
            current_sentence.append(wordSection["word"])
    return sentences

def second_to_timecode(x: float) -> str:
    hour, x = divmod(x, 3600)
    minute, x = divmod(x, 60)
    second, x = divmod(x, 1)
    millisecond = int(x * 1000.)

    return '%.2d:%.2d:%.2d,%.3d' % (hour, minute, second, millisecond)

def add_subtitles(video_path, subtitles_path, output_path):
    ffmpeg_command = [
        'ffmpeg',
        '-i', video_path,
        '-vf', f"subtitles={subtitles_path}",
        '-c:s', 'copy',
        output_path
    ]

    try:
        subprocess.run(ffmpeg_command, check=True)
        print("Subtitles added successfully!")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

def create_srt_file_new(lines, output_file):
    with open(output_file, 'w') as file:
        for i, line in enumerate(lines):
            #file.write(str(i + 1) + '\n')  # Subtitle number
            #file.write('00:00:00,000 --> 00:00:01,000\n')  # Subtitle timestamp (dummy values)
            file.write(line + '\n')  # Subtitle text
            file.write('\n')  # Blank line between subtitles
    print(f'SRT file "{output_file}" created successfully!')

def save_csv_file(data, filename):
  with open(filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=',')
    csv_writer.writerows(data)

def extract_videos_audios(folder_name,source_path,video,duration=30,step=25,starting_time=0):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    extracted_folder_path = folder_name+"extracted/"
    audio_folder_path = folder_name+"audios/"
    if not os.path.exists(extracted_folder_path):
        os.mkdir(extracted_folder_path)
    if not os.path.exists(audio_folder_path):
        os.mkdir(audio_folder_path)    
    # extract videos and audios from those videos
    starting_time = 0
    file_list = []
    for i in range(starting_time,int(video.duration),step):

        target1 = extracted_folder_path+str(i)+".mp4"
        # Check if file exists
        if not os.path.exists(target1):
            # Extract video
            ffmpeg_extract_subclip(source_path, i, i+duration, targetname=target1)

        audio_filename1 = audio_folder_path+str(i)+"_converted.wav"
        file_list.append(audio_filename1)
        # check if file exists
        if not os.path.exists(audio_filename1):
            # Extract audio
            video1 = VideoFileClip(target1)
            video1.audio.write_audiofile(audio_filename1)    


def transcribe_all_sentences(model,folder_name):
    # Transcribe the audios
    audio_folder_path = folder_name+"audios/"
    file_list = list_files_in_folder(audio_folder_path)

    list_all_sentences = dict()
    for inx , audioFile in enumerate(file_list):
        _ ,_ , sentence_timestamps =transcribe_test(model,audioFile,"de")
        print(sentence_timestamps)
        sentences = dict()
        if inx == 0:  # Ignore the last only
            sentences = extract_sentences(sentence_timestamps,extract_number(audioFile),False,True)
        elif inx == len(file_list)-1:  # Ignore the first only
            sentences = extract_sentences(sentence_timestamps,extract_number(audioFile),True,False)
        else: # Ignore both first and last sentences
            sentences = extract_sentences(sentence_timestamps,extract_number(audioFile),True,True)
        print(sentences)
        list_all_sentences.update(sentences)
    return list_all_sentences
import json
def save_dict_to_file(dictionary, file_path):
    with open(file_path, 'w') as file:
        json.dump(dictionary, file)

def read_dict_from_file(file_path):
    with open(file_path, 'r') as file:
        dictionary = json.load(file)
    return dictionary     

def list_files_in_folder(folder_path):
    file_names = []
    for file_name in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file_name)):
            file_names.append(folder_path+file_name)
    return file_names

def is_in_limits(value,min,max):
    if value <= max and value >=min:
        return True
    return False

def remove_spaces_before_punctuation(text):
    # Define a regular expression pattern to match spaces before punctuation
    pattern = r'\s+([.,!?])'
    
    # Replace spaces before punctuation with the punctuation mark
    modified_text = re.sub(pattern, r'\1', text)
    
    return modified_text
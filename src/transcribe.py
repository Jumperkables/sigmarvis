# standard imports
import argparse
import json
import os

# third party imports
from loguru import logger
import whisper
from tqdm import tqdm

# local imports
from data import data_utils

WHISPER_MODEL = "turbo" # "large" | "medium"
SAMPLING_RATE = 24000     # Standard for Tacotron2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transcribe audio files')
    parser.add_argument('-t', '--transcribe', action="store_true", help='Transcribe audio file')
    parser.add_argument('-c', '--coqui', action="store_true", help='Move files to Coqui TTS format')
    args = parser.parse_args()
    assert args.transcribe or args.coqui, "Please -t (--transcribe), and Then -c (--coqui)"


    if args.transcribe:
        ogg_files = data_utils.load_ogg_files()
        # for each of the ogg_files, rename any whitespace to underscores
        logger.info("Removing whitespace from ogg files...")
        for ogg_file in tqdm(ogg_files, total=len(ogg_files)):
            if ' ' in ogg_file:
                os.rename(ogg_file, ogg_file.replace(' ', '_'))

        logger.info("Converting ogg files to wav files...")
        # for each of the ogg_files, create a wav file
        for ogg_file in tqdm(ogg_files, total=len(ogg_files)):
            wav_file = ogg_file.replace('.ogg', '.wav')
            os.system(f"ffmpeg -i {ogg_file} -ar {SAMPLING_RATE} {wav_file}")

        logger.info("Transcribing audio files...")
        wav_files = [f.replace('.ogg', '.wav') for f in ogg_files]
        model = whisper.load_model(WHISPER_MODEL)
        for wav in tqdm(wav_files, total=len(wav_files)):
            transcript_file = wav.replace('.wav', '.json')
            if os.path.exists(transcript_file):
                print(f"Skipping {wav}")
                continue
            transcript = model.transcribe(wav)
            print(transcript['text'])
            with open(transcript_file, 'w') as f:
                json.dump(transcript, f)
    
    if args.coqui:
        logger.info("Formatting created wavs and transcripts to Coqui TTS format...")
        data_path = f"{data_utils.DATA_FOLDER}_{SAMPLING_RATE}"
        new_data_path = f"{data_path}_coqui-tts/wavs"
        # create a | separated metadata file with no headers
        os.makedirs(new_data_path)
        metadata_file = f"{data_path}_coqui-tts/metadata.txt"
        with open(metadata_file, 'w') as f:
            for root, dirs, files in os.walk(data_path):
                for file in files:
                    if file.endswith(".wav"):
                        # copy the wav file to the top level of the new_data_path
                        wav_file = os.path.join(root, file)
                        new_wav_file = os.path.join(new_data_path, file)
                        os.system(f"cp {wav_file} {new_wav_file}")

                        transcript_file = wav_file.replace('.wav', '.json')
                        with open(transcript_file, 'r') as f2:
                            transcript = json.load(f2)
                        string_to_write = f"{file.split('.wav')[0]}|{transcript['text']}|{transcript['text']}\n"
                        f.write(string_to_write)
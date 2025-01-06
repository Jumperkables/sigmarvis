import os, sys
import time
import json
import argparse

import speech_recognition as sr
from pydub import AudioSegment
from pydub.playback import play

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"

class MyRecognizer(sr.Recognizer):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Set up microphone
        devices = sr.Microphone.list_microphone_names()
        [ dev := idx for idx, device in enumerate(devices) if device.startswith("Scarlett Solo USB: Audio")]
        self.microphone = sr.Microphone(dev)

        # Set recognizer engine
        self.recognizer_switch = {
            "google": self.recognize_google,
            "sphinx": self.recognize_sphinx,
            "snowboy": self.snowboy_wait_for_hot_word,
        }

    def recognize(self, audio):
        return self.recognizer_switch[self.args.sr_engine](audio)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr_engine", default="sphinx", choices=["google", "sphinx", "snowboy"], help="Which speech recongition engine to use from the SpeechRecognition PyPi package")
    args = parser.parse_args()

    # Recognizer
    recognizer = MyRecognizer(args)

    # Ikit Diplomacy Lines
    ikit_dipl_audio_path = os.path.join(os.path.dirname(__file__), "data/ikit_diplomacy_lines/diplomacy_lines.wav")
    ikit_dipl_audio = AudioSegment.from_wav(ikit_dipl_audio_path)
    ikit_dipl_audio_rate = 1000 # PyDub works in milliseconds

    # Listen and repeat
    stop_signal = False
    while not stop_signal:
        # Recognise
        with recognizer.microphone as source:
            print("Say something!")
            audio = recognizer.listen(source)
        breakpoint()
        ret_str = recognizer.recognize(audio)
        print(f"{args.sr_engine} thinks you said: '{ret_str}'")
        
        # HANDLE THE RECOGNISED STRING
        ## Ikit responses
        ### what do you want from me
        if "hello" in ret_str.lower():
            start, end = 57.3*ikit_dipl_audio_rate, 60*ikit_dipl_audio_rate
            play(ikit_dipl_audio[start:end])
        ### scurry forth and squeak
        if ret_str.lower() in ["goodbye", "farewell"]:
            start, end = 247.3*ikit_dipl_audio_rate, 250*ikit_dipl_audio_rate
            play(ikit_dipl_audio[start:end])
            stop_signal = True
    print("\n\nSHUTTING DOWN\n")

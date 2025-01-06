import os, sys
import time
import json
import argparse
import time

import struct
import wave
import pyaudio
from pydub import AudioSegment
from pydub.playback import play

import torch
import torchaudio

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"




class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor):
        """Given a sequence emission over labels, get the best path
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          List[str]: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        joined = "".join([self.labels[i] for i in indices])
        return joined.replace("|", " ").strip().split()




class MyRecognizer():
    def __init__(self, args):
        self.args = args 
        p = pyaudio.PyAudio() # Get the microphone
        num_devices = p.get_host_api_info_by_index(0).get('deviceCount')
        mic = None
        [mic := device for device in range(num_devices) if p.get_device_info_by_host_api_device_index(0, device)['name'].startswith("Scarlett")]
        assert mic != None, f"The scarlett microphone has not been detected"
        del p
        self.mic = mic
        self.recorder = pyaudio.PyAudio()
        bundle = torchaudio.pipelines.HUBERT_ASR_LARGE
        self.recognizer = bundle.get_model()
        self.recognizer.eval()
        self.labels = bundle.get_labels()
        self.sample_rate = bundle.sample_rate
        self.ctc_decoder = GreedyCTCDecoder(self.labels, blank=0)
        ikit_dipl_audio_path = os.path.join(os.path.dirname(__file__), "data/ikit_diplomacy_lines/diplomacy_lines.wav")
        self.ikit_dipl_audio = AudioSegment.from_wav(ikit_dipl_audio_path)
        self.ikit_dipl_audio_rate = 1000 # PyDub works in milliseconds

    def play_ikit(self, start, end):
        play(self.ikit_dipl_audio[start*self.ikit_dipl_audio_rate:end*self.ikit_dipl_audio_rate])

    def record(self):
        CHUNK = 100
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        Y = 30
        stream = self.recorder.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK, input_device_index=self.mic)
        stream.start_stream()
        print("\n\nRecording until sound is too low...")
        frames=[]
        self.play_ikit(64,65)
        start_time = time.perf_counter()
        stop_recording = False
        while not(stop_recording): 
            data = stream.read(CHUNK, exception_on_overflow = False) # Converting chunk data into integers
            data_int = struct.unpack(str(2*CHUNK) +'B', data) 
            avg_data=sum(data_int)/len(data_int) # Finding average intensity per chunk
            sys.stdout.write("Avg: %d%%   \r" % (avg_data) )
            sys.stdout.flush()
            frames.append(data) # Recording chunk data
            if (avg_data < Y) and (time.perf_counter()-start_time)>=2: # Allow a 2 second grace period
                stop_recording = True
        stream.stop_stream()
        stream.close()
        print("Ending recording!")
        output_file = "/tmp/ikit_bot_recording.wav"
        if os.path.exists(output_file):
            os.remove(output_file)
        wf = wave.open(output_file, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.recorder.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        audio = b''.join(frames)
        wf.writeframes(audio)
        wf.close()
        waveform = torchaudio.load(output_file)
        return waveform

    def recognize(self, waveform):
        waveform = torchaudio.functional.resample(waveform[0], waveform[1], self.sample_rate)
        emissions, _ = self.recognizer(waveform)
        transcripts = self.ctc_decoder(emissions[0])
        return transcripts




class MyTTS():
    def __init__(self, args):
        self.args = args
        bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
        self.processor = bundle.get_text_processor()
        self.tacotron2 = bundle.get_tacotron2().to(args.device)
        self.tacotron2.eval()
        self.vocoder = bundle.get_vocoder().to(args.device)
        self.vocoder.eval()

    def placeholder(self, text):
        with torch.inference_mode():
            processed, lengths = self.processor(text)
            processed = processed.to(self.args.device)
            lengths = lengths.to(self.args.device)
            spec, spec_lengths, _ = self.tacotron2.infer(processed, lengths)
            waveforms, lengths = self.vocoder(spec, spec_lengths)
        torchaudio.save("/tmp/tts_output_wavernn.wav", waveforms[0:1].cpu(), sample_rate=self.vocoder.sample_rate)
        breakpoint()
        raise NotImplementedError("Reduce the volume and increase the actual quality")
        play(AudioSegment.from_wav("/tmp/tts_output_wavernn.wav"))




def process_command(command, stop_signal):
    command_str = " ".join(command)
    if "hello" in command: # what do you want from me
        recognizer.play_ikit(57.3, 60)
    elif command[0] == "farewell": # scurry forth and squeak
        recognizer.play_ikit(247.3, 250)
        stop_signal = True
    elif command == ['who', 'you', 'going', 'to', 'call'] or command == ['who', 'are', 'you', 'going', 'to', 'call']:
        recognizer.play_ikit(72.3, 74.6)
    return stop_signal


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument_group("General Arguments")
    parser.add_argument("--device", default=0, type=int, help="Which device to use. -1 = cpu")
    parser.add_argument_group("Debugging Arguments")
    parser.add_argument("--test_tts", action="store_true", help="Test the tts engine")
    args = parser.parse_args()
    args.device = torch.device(f"cuda:{args.device}" if (torch.cuda.is_available() and args.device > -1) else "cpu")

    if args.test_tts:
        tts = MyTTS(args)
        tts.placeholder("Read out this text")

    # Recognizer
    recognizer = MyRecognizer(args)

    # Listen and repeat
    #raise NotImplementedError("Run in the background, detect my hotword")
    stop_signal = False
    while not stop_signal:
        audio = recognizer.record()
        transcript = recognizer.recognize(audio)
        command = [word.lower() for word in transcript]
        print(command)  
        stop_signal = process_command(command, stop_signal)
    print("\n\nSHUTTING DOWN\n")

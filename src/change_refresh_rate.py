import os
from glob import glob
from tqdm import tqdm

def resample_wav_files(base_dir, sampling_rate=24000):
    """
    Recursively find and resample all .wav files in subdirectories.

    Args:
        base_dir (str): The root directory to start searching from.
        sampling_rate (int): The target sampling rate in Hz.
    """
    # Find all .wav files recursively
    wav_files = glob(os.path.join(base_dir, "**", "*.wav"), recursive=True)
    
    # Use tqdm to show a progress bar
    breakpoint()
    for wav_file in tqdm(wav_files, desc="Resampling .wav files", unit="file"):
        # Temporary output file to avoid overwriting during processing
        temp_file = f"tmp.wav"

        # Construct the ffmpeg command with quotes around filenames
        command = f'ffmpeg -i {wav_file} -ar {sampling_rate} -ac 1 {temp_file}'

        # Run the resampling command
        os.system(command)

        # Replace original file with resampled file
        os.replace(temp_file, wav_file)  # Overwrite the original .wav file

# Set the base directory and sampling rate
base_directory = "/home/jumperkables/projects/voice_assistant/data/_Saltzpyre_1bd6b13e01e224f5.stream_24000"
target_sampling_rate = 24000

# Call the function
resample_wav_files(base_directory, target_sampling_rate)
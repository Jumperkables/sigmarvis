# Sigmarvis: Faith, Silicone, and Transcription
Voice assistant that I finetuned with Victor Saltzpyre's voice to run in the background and pick up requests for information.

## Why Victor Saltzpyre?
I will overlook your heresy if you study [the sacred texts.](https://www.youtube.com/watch?v=KRLs4xH48Wg)

## Requirements:
- To use coqui-ai/TTS => 3.9 =< python < 3.12

## Installation:
1. Gather the data
    * This [Reddit post](https://www.reddit.com/r/Vermintide/comments/pmrw0g/updated_audio_files_rip_voice_lines_sound_effects/) claims to have been given permission to scrape the data by a FatShark employee.
    * [Its from this google drive](https://drive.google.com/drive/folders/1GmG7Il91MJ2fbT9VrvryCGf3xzfALAbL)
    * Download only _Saltzpyre (1b...) file. This should be a 7zip
    * Extract the 7zip file, and name the resulting folder `_Saltzpyre_1bd6b13e01e224f5.stream`. Place this in `$ROOT/data/`. i.e. `$ROOT/data/_Saltzpyre_1bd6b13e01e224f5.stream/` should contain lots of various subfolders of ogg files
    * Manually rename any whitespaces in **folder** names to underscores. The script takes care of the **files**.
2. `pip install -r requirements.txt`
3. Further install [coqui TTS](https://github.com/coqui-ai/TTS)
    * `cd /path/to/your/git/downloads`
    * `git clone git@github.com:coqui-ai/TTS.git`
    * `pip install -e .`
4. Prepare the training data:
    * You will need to pair the audio files with their transcriptions. For legal reasons, I will not provide the transcriptions.
    * Don't worry, you'll generate your own, and go through and clean them up using `$ROOT/src/transcribe.py`
    * `python -m ./src/transcribe.py -t` will generate a transcription file for each audio file in the data directory. This will be a good starting point for cleaning up the transcriptions.
    * You may need to manually rename files with ' ' in to underscores

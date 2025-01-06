import os
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

print("Loading model...")
config = XttsConfig()
config.load_json("/home/jumperkables/projects/sigmarvis/src/run/training/GPT_XTTS_v2.0_LJSpeech_FT-January-06-2025_12+13AM-0000000/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_path="/home/jumperkables/projects/sigmarvis/src/run/training/GPT_XTTS_v2.0_LJSpeech_FT-January-06-2025_12+13AM-0000000/best_model_47580.pth", use_deepspeed=False)
model.cuda()

print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
    audio_path=["/home/jumperkables/projects/sigmarvis/data/_Saltzpyre_1bd6b13e01e224f5.stream_22050_coqui-tts/wavs/thats_the_tower_or_im_a_heretic.wav"]
)

print("Inference...")
out = model.inference(
    "This jumpercables fellow has trained me to interface with the large language model Llama three for a home assistant bundle, and lets me judge the heretics for Sigmar in my own time. Lucky me.",
    "en",
    gpt_cond_latent,
    speaker_embedding,
    temperature=0.7, # Add custom parameters here
)
torchaudio.save("ex1.wav", torch.tensor(out["wav"]).unsqueeze(0), 24000)
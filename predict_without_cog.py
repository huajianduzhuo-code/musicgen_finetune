import os
import random
import subprocess
import typing as tp
import numpy as np
from typing import Optional
from pathlib import Path

import torch
import torchaudio
from omegaconf import OmegaConf

from audiocraft.models import MusicGen, MultiBandDiffusion
from audiocraft.solvers.compression import CompressionSolver
from audiocraft.models.loaders import load_compression_model, load_lm_model
from audiocraft.data.audio import audio_write
from audiocraft.models.builders import get_lm_model, get_compression_model, get_wrapped_compression_model

MODEL_PATH = "my_models"
os.environ["TRANSFORMERS_CACHE"] = MODEL_PATH
os.environ["TORCH_HOME"] = MODEL_PATH

def _delete_param(cfg, full_name: str):
    parts = full_name.split('.')
    for part in parts[:-1]:
        if part in cfg:
            cfg = cfg[part]
        else:
            return
    OmegaConf.set_struct(cfg, False)
    if parts[-1] in cfg:
        del cfg[parts[-1]]
    OmegaConf.set_struct(cfg, True)

def load_ckpt(path, device):
    # loaded = torch.hub.load_state_dict_from_url(str(path))
    loaded = torch.load(str(path))
    cfg = OmegaConf.create(loaded['xp.cfg'])
    cfg.device = str(device)
    if cfg.device == 'cpu':
        cfg.dtype = 'float32'
    else:
        cfg.dtype = 'float16'
    _delete_param(cfg, 'conditioners.self_wav.chroma_chord.cache_path')
    _delete_param(cfg, 'conditioners.self_wav.chroma_stem.cache_path')
    _delete_param(cfg, 'conditioners.args.merge_text_conditions_p')
    _delete_param(cfg, 'conditioners.args.drop_desc_p')

    lm = get_lm_model(loaded['xp.cfg'])
    lm.load_state_dict(loaded['model']) 
    lm.eval()
    lm.cfg = cfg
    compression_model = CompressionSolver.model_from_checkpoint(cfg.compression_model_checkpoint, device=device)
    return MusicGen(f"{os.getenv('COG_USERNAME')}/musicgen-finetuned", compression_model, lm)

class Predictor:
    def setup(self, weights: Optional[Path] = None, use_trained=False):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.mbd = MultiBandDiffusion.get_mbd_musicgen()

        if str(weights) == "weights":
            weights = None

        self.tuned_weights = None

        if not use_trained:
            self.model = self._load_model(
                model_path=MODEL_PATH,
                cls=MusicGen,
                model_id="facebook/musicgen-stereo-melody",
            )

    def _load_model(
        self,
        model_path: str,
        cls: Optional[any] = None,
        load_args: Optional[dict] = {},
        model_id: Optional[str] = None,
        device: Optional[str] = None,
    ) -> MusicGen:

        if device is None:
            device = self.device

        compression_model = load_compression_model(
            model_id, device=device, cache_dir=model_path
        )
        lm = load_lm_model(model_id, device=device, cache_dir=model_path)

        return MusicGen(model_id, compression_model, lm)

    def predict(
        self,
        prompt: str = None,
        input_audio: Path = None,
        duration: int = 8,
        continuation: bool = False,
        continuation_start: int = 0,
        continuation_end: int = None,
        multi_band_diffusion: bool = False,
        normalization_strategy: str = "loudness",
        top_k: int = 250,
        top_p: float = 0.0,
        temperature: float = 1.0,
        classifier_free_guidance: int = 3,
        output_format: str = "wav",
        seed: int = None,
        replicate_weights: str = None,
    ) -> Path:

        if prompt is None and input_audio is None:
            raise ValueError("Must provide either prompt or input_audio")
        if continuation and not input_audio:
            raise ValueError("Must provide `input_audio` if continuation is `True`.")

        if replicate_weights and replicate_weights != self.tuned_weights:
            print("LOADING.........")
            self.model = load_ckpt(replicate_weights, self.device)
            print("Fine-tuned model weights hot-swapped!")
            self.tuned_weights = replicate_weights

        if multi_band_diffusion and int(self.model.lm.cfg.transformer_lm.n_q) == 8:
            raise ValueError("Multi-band Diffusion only works with non-stereo models.")

        model = self.model

        set_generation_params = lambda duration: model.set_generation_params(
            duration=duration,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            cfg_coef=classifier_free_guidance,
        )
        
        if not seed or seed == -1:
            seed = torch.seed() % 2 ** 32 - 1
            set_all_seeds(seed)
        set_all_seeds(seed)
        print(f"Using seed {seed}")

        if not input_audio:
            set_generation_params(duration)
            wav, tokens = model.generate([prompt], progress=True, return_tokens=True)

        else:
            input_audio, sr = torchaudio.load(input_audio)
            input_audio = input_audio[None] if input_audio.dim() == 2 else input_audio

            continuation_start = 0 if not continuation_start else continuation_start
            if continuation_end is None or continuation_end == -1:
                continuation_end = input_audio.shape[2] / sr

            if continuation_start > continuation_end:
                raise ValueError(
                    "`continuation_start` must be less than or equal to `continuation_end`"
                )

            input_audio_wavform = input_audio[
                ..., int(sr * continuation_start) : int(sr * continuation_end)
            ]
            input_audio_duration = input_audio_wavform.shape[-1] / sr

            if continuation:
                set_generation_params(duration)
                wav, tokens = model.generate_continuation(
                    prompt=input_audio_wavform,
                    prompt_sample_rate=sr,
                    descriptions=[prompt],
                    progress=True, 
                    return_tokens=True,
                )

            else:
                set_generation_params(duration)
                wav, tokens = model.generate_with_chroma(
                    [prompt], input_audio_wavform, sr, progress=True, return_tokens=True
                )
        if multi_band_diffusion:
            wav = self.mbd.tokens_to_wav(tokens)

        if replicate_weights is not None:
            model_save_folder = f"./generation_output/{str(replicate_weights)}"
            os.makedirs(model_save_folder, exist_ok=True)
        else:
            model_save_folder = "./generation_output/original_model"
            os.makedirs(model_save_folder, exist_ok=True)
        
        file_name = '_'.join(prompt.split(" "))

        audio_write(
            f"{model_save_folder}/{file_name}",
            wav[0].cpu(),
            model.sample_rate,
            strategy=normalization_strategy,
        )
        wav_path = "out.wav"

        if output_format == "mp3":
            mp3_path = "out.mp3"
            if Path(mp3_path).exists():
                os.remove(mp3_path)
            subprocess.call(["ffmpeg", "-i", wav_path, mp3_path])
            os.remove(wav_path)
            path = mp3_path
        else:
            path = wav_path

        return Path(path)

def set_all_seeds(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    predictor = Predictor()
    predictor.setup(use_trained=True)

    output_path = predictor.predict(
        prompt="guoquyinyue. piano and string. excited.",
        duration=60,
        replicate_weights='trained_model/guoqu3epoch100iter.th'
    )
    print(f"Generated audio saved at {output_path}")

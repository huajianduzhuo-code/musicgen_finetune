import logging
import os
import subprocess
from tqdm import tqdm
import shutil
import torch
import json
import numpy as np
import librosa
from pathlib import Path
from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs, TensorflowPredict2D

# from essentia.streaming import TensorflowPredictEffnetDiscogs, TensorflowPredict2D
# from essentia.standard import MonoLoader, TensorflowPredict2D

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)
logging.getLogger("py4j").setLevel(logging.WARNING)
logging.getLogger("sh.command").setLevel(logging.ERROR)

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'


def prepare_data(
        dataset_path,
        target_path='src/train_data',
        one_same_description=None,
        meta_path='src/meta',
        auto_labeling=True,
        drop_vocals=True,
        device='cuda',
        channels=2):

    d_path = Path(target_path)
    d_path.mkdir(exist_ok=True, parents=True)

    # Decompressing file at dataset_path
    if str(dataset_path).rsplit('.', 1)[1] == 'zip':
        subprocess.run(['unzip', str(dataset_path), '-d', target_path + '/'])
    elif str(dataset_path).rsplit('.', 1)[1] == 'tar':
        subprocess.run(['tar', '-xvf', str(dataset_path), '-C', target_path + '/'])
    elif str(dataset_path).rsplit('.', 1)[1] == 'gz':
        subprocess.run(['tar', '-xvzf', str(dataset_path), '-C', target_path + '/'])
    elif str(dataset_path).rsplit('.', 1)[1] == 'tgz':
        subprocess.run(['tar', '-xzvf', str(dataset_path), '-C', target_path + '/'])
    elif str(dataset_path).rsplit('.', 1)[1] in ['wav', 'mp3', 'flac', 'mp4']:
        shutil.move(str(dataset_path), target_path + '/' + str(dataset_path.name))
    else:
        raise Exception("Not supported compression file type. The file type should be one of 'zip', 'tar', 'tar.gz', 'tgz' types of compression file, or a single 'wav', 'mp3', 'flac', 'mp4' types of audio file.")

    # Removing __MACOSX and .DS_Store
    if (Path(target_path)/"__MACOSX").is_dir():
        shutil.rmtree(target_path+"/__MACOSX")
    elif (Path(target_path)/"__MACOSX").is_file():
        os.remove(target_path+"/__MACOSX")
    if (Path(target_path)/".DS_Store").is_dir():
        shutil.rmtree(target_path+"/.DS_Store")
    elif (Path(target_path)/".DS_Store").is_file():
        os.remove(target_path+"/.DS_Store")

    # Audio Chunking and Vocal Dropping
    from pydub import AudioSegment

    if drop_vocals:
        import demucs.pretrained
        import torchaudio
        separator = demucs.pretrained.get_model('mdx_extra').to('cuda')
    else:
        separator = None

    for filename in tqdm(os.listdir(target_path)):
        if filename.endswith(('.mp3', '.wav', '.flac', '.mp4')):
            if filename.endswith(('.mp4')):
                import moviepy.editor
                video = moviepy.editor.VideoFileClip(os.path.join(target_path, filename))
                fname = filename.rsplit('.', 1)[0] + '.wav'
                video.audio.write_audiofile(os.path.join(target_path, fname))
                print(f'A mp4 file is converted into a wav file : {filename}')
                os.remove(target_path + '/' + filename)
            else:
                fname = filename
            # Chuking audio files into 30sec chunks
            audio = AudioSegment.from_file(target_path + '/' + fname)

            audio = audio.set_frame_rate(44100)  # Resampling to 44100

            if len(audio) > 30000:
                print('Chunking ' + fname)

                # Splitting the audio files into 30-second chunks
                for i in range(0, len(audio), 30000):
                    chunk = audio[i:i + 30000]
                    if len(chunk) > 5000:  # Omitting residuals with <5sec duration
                        if drop_vocals and separator is not None:
                            from demucs.apply import apply_model
                            from demucs.audio import convert_audio
                            print('Separating Vocals from ' + f"{target_path + '/' + fname[:-4]}_chunk{i//1000}.wav")

                            channel_sounds = chunk.split_to_mono()
                            samples = [s.get_array_of_samples() for s in channel_sounds]

                            chunk = np.array(samples).T.astype(np.float32)
                            chunk /= np.iinfo(samples[0].typecode).max
                            chunk = torch.Tensor(chunk).T
                            print(chunk.shape)

                            # Resample for Demucs
                            chunk = convert_audio(chunk, 44100, separator.samplerate, separator.audio_channels)
                            stems = apply_model(separator, chunk[None], device='cuda')
                            stems = stems[:, [separator.sources.index('bass'), separator.sources.index('drums'), separator.sources.index('other')]]
                            mixed = stems.sum(1)
                            torchaudio.save(f"{target_path + '/' + fname[:-4]}_chunk{i//1000}.wav", mixed.squeeze(0), separator.samplerate)
                        else:
                            chunk.export(f"{target_path + '/' + fname[:-4]}_chunk{i//1000}.wav", format="wav")
                os.remove(target_path + '/' + fname)

    max_sample_rate = 0

    # Auto Labeling
    if auto_labeling:
        sp.call(["curl", "https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb", "--output", "genre_discogs400-discogs-effnet-1.pb"])
        sp.call(["curl", "https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb", "--output", "discogs-effnet-bs64-1.pb"])
        sp.call(["curl", "https://essentia.upf.edu/models/classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-discogs-effnet-1.pb", "--output", "mtg_jamendo_moodtheme-discogs-effnet-1.pb"])
        sp.call(["curl", "https://essentia.upf.edu/models/classification-heads/mtg_jamendo_instrument/mtg_jamendo_instrument-discogs-effnet-1.pb", "--output", "mtg_jamendo_instrument-discogs-effnet-1.pb"])

        from metadata import genre_labels, mood_theme_classes, instrument_classes

        def filter_predictions(predictions, class_list, threshold=0.1):
            predictions_mean = np.mean(predictions, axis=0)
            sorted_indices = np.argsort(predictions_mean)[::-1]
            filtered_indices = [i for i in sorted_indices if predictions_mean[i] > threshold]
            filtered_labels = [class_list[i] for i in filtered_indices]
            filtered_values = [predictions_mean[i] for i in filtered_indices]
            return filtered_labels, filtered_values

        def make_comma_separated_unique(tags):
            seen_tags = set()
            result = []
            for tag in ', '.join(tags).split(', '):
                if tag not in seen_tags:
                    result.append(tag)
                    seen_tags.add(tag)
            return ', '.join(result)

        def get_audio_features(audio_filename):
            audio = MonoLoader(filename=audio_filename, sampleRate=16000, resampleQuality=4)()
            embedding_model = TensorflowPredictEffnetDiscogs(graphFilename="discogs-effnet-bs64-1.pb", output="PartitionedCall:1")
            # embedding_model = TensorflowPredict2D(graphFilename="discogs-effnet-bs64-1.pb", output="PartitionedCall:1")
            embeddings = embedding_model(audio)

            result_dict = {}

            # Predicting genres
            genre_model = TensorflowPredict2D(graphFilename="genre_discogs400-discogs-effnet-1.pb", input="serving_default_model_Placeholder", output="PartitionedCall:0")
            predictions = genre_model(embeddings)
            filtered_labels, _ = filter_predictions(predictions, genre_labels)
            filtered_labels = ', '.join(filtered_labels).replace("---", ", ").split(', ')
            result_dict['genres'] = make_comma_separated_unique(filtered_labels)

            # Predicting mood/theme
            mood_model = TensorflowPredict2D(graphFilename="mtg_jamendo_moodtheme-discogs-effnet-1.pb")
            predictions = mood_model(embeddings)
            filtered_labels, _ = filter_predictions(predictions, mood_theme_classes, threshold=0.05)
            result_dict['moods'] = make_comma_separated_unique(filtered_labels)

            # Predicting instruments
            instrument_model = TensorflowPredict2D(graphFilename="mtg_jamendo_instrument-discogs-effnet-1.pb")
            predictions = instrument_model(embeddings)
            filtered_labels, _ = filter_predictions(predictions, instrument_classes)
            result_dict['instruments'] = filtered_labels

            return result_dict

        train_len = 0

        os.mkdir(meta_path)
        with open(meta_path + "/data.jsonl", "w") as train_file:
            files = list(d_path.rglob('*.mp3')) + list(d_path.rglob('*.wav')) + list(d_path.rglob('*.flac'))
            if len(files) == 0:
                raise ValueError("No audio file detected. Are you sure the audio file is longer than 5 seconds?")
            for filename in tqdm(files):
                result = get_audio_features(str(filename))

                # Obtaining key and BPM
                y, sr = librosa.load(str(filename))
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                tempo = round(tempo)
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                key = np.argmax(np.sum(chroma, axis=1))
                key = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][key]
                length = librosa.get_duration(y=y, sr=sr)

                sr = librosa.get_samplerate(str(filename))
                if sr > max_sample_rate:
                    max_sample_rate = sr

                artist_name = ""

                entry = {
                    "key": f"{key}",
                    "artist": artist_name,
                    "sample_rate": sr,
                    "file_extension": "wav",
                    "description": "",
                    "keywords": "",
                    "duration": length,
                    "bpm": tempo,
                    "genre": result.get('genres', ""),
                    "title": "",
                    "name": "",
                    "instrument": result.get('instruments', ""),
                    "moods": result.get('moods', []),
                    "path": str(filename),
                }
                with open(str(filename).rsplit('.', 1)[0] + '.json', "w") as file:
                    json.dump(entry, file)
                print(entry)

                train_len += 1
                train_file.write(json.dumps(entry) + '\n')

            from numba import cuda
            device = cuda.get_current_device()
            device.reset()

            filelen = len(files)
    else:
        import audiocraft.data.audio_dataset

        meta = audiocraft.data.audio_dataset.find_audio_files(target_path, audiocraft.data.audio_dataset.DEFAULT_EXTS, progress=True, resolve=False, minimal=True, workers=10)

        if len(meta) == 0:
            raise ValueError("No audio file detected. Are you sure the audio file is longer than 5 seconds?")

        for m in meta:
            if m.sample_rate > max_sample_rate:
                max_sample_rate = m.sample_rate
            fdict = {
                "key": "",
                "artist": "",
                "sample_rate": m.sample_rate,
                "file_extension": m.path.rsplit('.', 1)[1],
                "description": "",
                "keywords": "",
                "duration": m.duration,
                "bpm": "",
                "genre": "",
                "title": "",
                "name": Path(m.path).name.rsplit('.', 1)[0],
                "instrument": "",
                "moods": []
            }
            with open(m.path.rsplit('.', 1)[0] + '.json', "w") as file:
                json.dump(fdict, file)
        audiocraft.data.audio_dataset.save_audio_meta(meta_path + '/data.jsonl', meta)
        filelen = len(meta)

    audios = list(d_path.rglob('*.mp3')) + list(d_path.rglob('*.wav'))

    for audio in list(audios):
        jsonf = open(str(audio).rsplit('.', 1)[0] + '.json', 'r')
        fdict = json.load(jsonf)
        jsonf.close()

        if one_same_description is None:
            if Path(str(audio).rsplit('.', 1)[0] + '.txt').exists():
                f = open(str(audio).rsplit('.', 1)[0] + '.txt', 'r')
                line = f.readline()
                f.close()
                fdict["description"] = line
            elif Path(str(audio).rsplit('_chunk', 1)[0] + '.txt').exists():
                f = open(str(audio).rsplit('_chunk', 1)[0] + '.txt', 'r')
                line = f.readline()
                f.close()
                fdict["description"] = line
        else:
            fdict["description"] = one_same_description

        with open(str(audio).rsplit('.', 1)[0] + '.json', "w") as file:
            json.dump(fdict, file)

    return max_sample_rate, filelen


def train(
        dataset_path,
        auto_labeling=True,
        drop_vocals=True,
        one_same_description=None,
        model_version="stereo-melody",
        epochs=3,
        updates_per_epoch=100,
        batch_size=16,
        optimizer='dadam',
        lr=1.0,
        lr_scheduler="cosine",
        warmup=8,
        cfg_p=0.3):
    meta_path = 'src/meta'
    target_path = 'src/train_data'

    out_path = "trained_model.tar"

    # Removing previous training's leftover
    if os.path.isfile(out_path):
        os.remove(out_path)
    if os.path.isfile('weights'):
        os.remove('weights')
    if os.path.isfile('weight'):
        os.remove('weight')
    if os.path.isdir('weights'):
        shutil.rmtree('weights')
    if os.path.isdir('weight'):
        shutil.rmtree('weight')
    if os.path.isdir(meta_path):
        shutil.rmtree(meta_path)
    if os.path.isdir(target_path):
        shutil.rmtree(target_path)
    if os.path.isdir('models_backup'):
        shutil.rmtree('models_backup')
    if os.path.isdir('tmp'):
        shutil.rmtree('tmp')

    if "stereo" in model_version:
        channels = 2
    else:
        channels = 1

    print("target_path", target_path)
    print("dataset_path", dataset_path)
    print("one_same_description", one_same_description)
    max_sample_rate, len_dataset = prepare_data(dataset_path, target_path, one_same_description, meta_path, auto_labeling, drop_vocals, 'cuda', channels)

    # if model_version in ["melody", "stereo-melody", "medium", "stereo-medium"]:
    #     batch_size = 8
    #     print(f"Batch size is reset to {batch_size}, since `medium(melody)` model can only be trained with 8 with current GPU settings.")

    # if batch_size % 8 != 0:
    #     batch_size = batch_size - (batch_size % 8)
    #     print(f"Batch size is reset to {batch_size}, the multiple of 8(number of gpus).")
    batch_size = 1

    # Setting up dora args
    if model_version not in ["melody", "stereo-melody"]:
        solver = "musicgen/musicgen_base_32khz"
        if "stereo" in model_version:
            model_scale = model_version.rsplit('-')[-1]
        else:
            model_scale = model_version
        conditioner = "text2music"
    else:
        solver = "musicgen/musicgen_melody_32khz"
        model_scale = "medium"
        conditioner = "chroma2music"
    continue_from = f"//pretrained/facebook/musicgen-{model_version}"

    args = ["run", "-d", "--", f"solver={solver}", f"model/lm/model_scale={model_scale}", f"continue_from={continue_from}", f"conditioner={conditioner}"]
    if "stereo" in model_version:
        args.append(f"codebooks_pattern.delay.delays={[0, 0, 1, 1, 2, 2, 3, 3]}")
        args.append('transformer_lm.n_q=8')
        args.append('interleave_stereo_codebooks.use=True')
        args.append('channels=2')
    args.append(f"datasource.max_sample_rate={max_sample_rate}")
    args.append(f"datasource.train={meta_path}")
    args.append(f"dataset.train.num_samples={len_dataset}")
    args.append(f"optim.epochs={epochs}")
    args.append(f"optim.lr={lr}")
    args.append(f"schedule.lr_scheduler={lr_scheduler}")
    args.append(f"schedule.cosine.warmup={warmup}")
    args.append(f"schedule.polynomial_decay.warmup={warmup}")
    args.append(f"schedule.inverse_sqrt.warmup={warmup}")
    args.append(f"schedule.linear_warmup.warmup={warmup}")
    args.append(f"classifier_free_guidance.training_dropout={cfg_p}")
    if updates_per_epoch is not None:
        args.append(f"logging.log_updates={updates_per_epoch // 10 if updates_per_epoch // 10 >= 1 else 1}")
    else:
        args.append(f"logging.log_updates=0")
    args.append(f"dataset.batch_size={batch_size}")
    args.append(f"optim.optimizer={optimizer}")

    if updates_per_epoch is None:
        args.append("dataset.train.permutation_on_files=False")
        args.append("optim.updates_per_epoch=1")
    else:
        args.append("dataset.train.permutation_on_files=True")
        args.append(f"optim.updates_per_epoch={updates_per_epoch}")

    print("到这里了")
    print(["dora"] + args)

    subprocess.call(["dora"] + args)

    print("遍历根目录开始")
    for dirpath, dirnames, filenames in os.walk("tmp"):
        print("遍历根目录")
        for filename in [f for f in filenames if f == "checkpoint.th"]:
            checkpoint_dir = os.path.join(dirpath, filename)
    
    print(checkpoint_dir)
    loaded = torch.load(checkpoint_dir, map_location=torch.device('cpu'))

    torch.save({'xp.cfg': loaded["xp.cfg"], "model": loaded["model"]}, out_path)

    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a music generation model.")
    parser.add_argument("dataset_path", type=Path, help="Path to dataset directory.")
    parser.add_argument("--auto_labeling", action="store_true", help="Enable auto-labeling.")
    parser.add_argument("--drop_vocals", action="store_true", help="Drop vocals from audio files.")
    parser.add_argument("--one_same_description", type=str, default=None, help="A description for all of audio data.")
    parser.add_argument("--model_version", type=str, default="stereo-melody", choices=["stereo-melody", "stereo-small", "stereo-medium", "melody", "small", "medium"], help="Model version to train.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train for.")
    parser.add_argument("--updates_per_epoch", type=int, default=100, help="Number of iterations for one epoch.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size. Must be multiple of 8 for 8-gpu training.")
    parser.add_argument("--optimizer", type=str, default='dadam', choices=["dadam", "adamw"], help="Type of optimizer.")
    parser.add_argument("--lr", type=float, default=1.0, help="Learning rate.")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["exponential", "cosine", "polynomial_decay", "inverse_sqrt", "linear_warmup"], help="Type of lr_scheduler.")
    parser.add_argument("--warmup", type=int, default=8, help="Warmup of lr_scheduler.")
    parser.add_argument("--cfg_p", type=float, default=0.3, help="CFG dropout ratio.")

    args = parser.parse_args()

    weights_path = train(
        dataset_path=args.dataset_path,
        auto_labeling=args.auto_labeling,
        drop_vocals=args.drop_vocals,
        one_same_description=args.one_same_description,
        model_version=args.model_version,
        epochs=args.epochs,
        updates_per_epoch=args.updates_per_epoch,
        batch_size=args.batch_size,
        optimizer=args.optimizer,
        lr=args.lr,
        lr_scheduler=args.lr_scheduler,
        warmup=args.warmup,
        cfg_p=args.cfg_p
    )

    print(f"Training completed. Weights saved at {weights_path}")

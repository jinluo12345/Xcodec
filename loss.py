import os
import numpy as np
import librosa
import soundfile as sf
import argparse
import glob
import logging
from pesq import pesq
from scipy import signal
from scipy.linalg import norm
from tqdm import tqdm
from pystoi import stoi
from stopes.eval.vocal_style_similarity.vocal_style_sim_tool import get_embedder, compute_cosine_similarity
import torch
from torchaudio.transforms import MelSpectrogram
import torch.nn as nn

# Mel Spectrogram Loss Calculation
class MelSpectrogramLoss(nn.Module):
    def __init__(
        self,
        n_mels_list: list = [80, 150],
        window_lengths: list = [2048, 512],
        loss_fn: nn.Module = nn.L1Loss(),
        mag_weight: float = 1.0,
        log_weight: float = 1.0,
        clamp_eps: float = 1e-5,
        pow: float = 2.0,
    ):
        super(MelSpectrogramLoss, self).__init__()
        self.n_mels_list = n_mels_list
        self.window_lengths = window_lengths
        self.loss_fn = loss_fn
        self.mag_weight = mag_weight
        self.log_weight = log_weight
        self.clamp_eps = clamp_eps
        self.pow = pow
        self.mel_transformers = [
            MelSpectrogram(
                sample_rate=16000,
                n_mels=n_mels,
                hop_length=window_length // 4,
                n_fft=window_length,
            ) for n_mels, window_length in zip(n_mels_list, window_lengths)
        ]

    def forward(self, ref, deg):
        loss = 0.0
        for mel_transformer, n_mels in zip(self.mel_transformers, self.n_mels_list):
            ref_mel = mel_transformer(ref)
            deg_mel = mel_transformer(deg)

            ref_mel_log = torch.log10(ref_mel.clamp(min=self.clamp_eps)).pow(self.pow)
            deg_mel_log = torch.log10(deg_mel.clamp(min=self.clamp_eps)).pow(self.pow)

            loss += self.mag_weight * self.loss_fn(ref_mel, deg_mel)
            loss += self.log_weight * self.loss_fn(ref_mel_log, deg_mel_log)
        
        return loss

def calculate_melspec_loss(ref_list, deg_list):
    mel_loss_fn = MelSpectrogramLoss()
    losses = []
    
    for ref, deg in zip(ref_list, deg_list):
        ref_tensor = torch.tensor(ref[np.newaxis, np.newaxis, :]).float()
        deg_tensor = torch.tensor(deg[np.newaxis, np.newaxis, :]).float()
        loss = mel_loss_fn(ref_tensor, deg_tensor)
        losses.append(loss.item())
    
    return losses

def find_audio_files(input_dir):
    audio_extensions = ['*.flac', '*.mp3', '*.wav']
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(input_dir, '**', ext), recursive=True))
    return sorted(audio_files)

def evaluate_speaker_similarity(ref_list, syn_list, model_path):
    embedder = get_embedder(model_name="valle", model_path=model_path)
    src_embs = embedder(ref_list)
    tgt_embs = embedder(syn_list)
    similarities = compute_cosine_similarity(src_embs, tgt_embs)
    return similarities, np.mean(similarities)

def evaluate(
    ref_folder=None,
    est_folder=None,
    metrics=None,
    ref_files=None,
    est_files=None,
    target_sr=16000,
    model_path='/remote-home1/share/personal/ytgong/wavlm_large_finetune.pth'
):
    if metrics is None:
        metrics = []
    if not metrics:
        raise ValueError("At least one metric must be specified.")

    valid_metrics = {'pesq-nb', 'pesq-wb', 'stoi', 'speakersim', 'melspec-loss'}
    for metric in metrics:
        if metric not in valid_metrics:
            raise ValueError(f"Invalid metric '{metric}'. Supported: {valid_metrics}")

    ref_files = ref_files or find_audio_files(ref_folder)
    est_files = est_files or find_audio_files(est_folder)

    ref_files = sorted(ref_files)
    est_files = sorted(est_files)

    # Check if the number of reference and estimated files match
    if len(ref_files) != len(est_files):
        raise ValueError("Number of reference and estimated files must match.")

    # Load all reference and estimated audio files
    ref_audios = []
    est_audios = []
    ref_srs = []
    est_srs = []

    for ref_path, est_path in zip(ref_files, est_files):
        assert os.path.basename(ref_path) == os.path.basename(est_path), "Reference and estimated audio filenames do not match!"
        ref, ref_sr = sf.read(ref_path)
        est, est_sr = sf.read(est_path)
        ref_audios.append(ref)
        est_audios.append(est)
        ref_srs.append(ref_sr)
        est_srs.append(est_sr)

    # Resample all audio to target_sr
    ref_audios_processed = []
    est_audios_processed = []
    for ref, ref_sr, est, est_sr in zip(ref_audios, ref_srs, est_audios, est_srs):
        if ref_sr != target_sr:
            ref = librosa.resample(ref.astype(np.float32), orig_sr=ref_sr, target_sr=target_sr)
        if est_sr != target_sr:
            est = librosa.resample(est.astype(np.float32), orig_sr=est_sr, target_sr=target_sr)

        min_len = min(len(ref), len(est))
        ref_processed = ref[:min_len].flatten()
        est_processed = est[:min_len].flatten()

        ref_audios_processed.append(ref_processed)
        est_audios_processed.append(est_processed)

    results = {metric: {'scores': [], 'mean': None} for metric in metrics}

    # Calculate metrics in batches
    if 'pesq-nb' in metrics or 'pesq-wb' in metrics:
        pesq_scores = {}
        for mode in ['nb', 'wb']:
            metric_key = f'pesq-{mode}'
            if metric_key in metrics:
                scores = []
                for ref, est in zip(ref_audios_processed, est_audios_processed):
                    score = pesq(target_sr, ref, est, mode=mode)
                    scores.append(score)
                pesq_scores[metric_key] = scores

    if 'stoi' in metrics:
        stoi_scores = []
        for ref, est in zip(ref_audios_processed, est_audios_processed):
            score = stoi(ref, est, target_sr)
            stoi_scores.append(score)

    if 'speakersim' in metrics:
        similarities, _ = evaluate_speaker_similarity(ref_files, est_files, model_path)

    if 'melspec-loss' in metrics:
        melspec_losses = calculate_melspec_loss(ref_audios_processed, est_audios_processed)

    # Populate results dictionary
    for metric in metrics:
        try:
            if metric.startswith('pesq-'):
                mode = metric.split('-')[1]
                results[metric]['scores'] = pesq_scores[metric]
            elif metric == 'stoi':
                results['stoi']['scores'] = stoi_scores
            elif metric == 'speakersim':
                results['speakersim']['scores'] = similarities
            elif metric == 'melspec-loss':
                results['melspec-loss']['scores'] = melspec_losses
        except KeyError:
            logging.error(f"No scores calculated for metric {metric}.")
            results[metric]['scores'] = [np.nan] * len(ref_files)

    # Compute means
    for metric in metrics:
        valid_scores = [s for s in results[metric]['scores'] if not np.isnan(s)]
        results[metric]['mean'] = np.mean(valid_scores) if valid_scores else np.nan

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate audio quality metrics.")
    parser.add_argument('-r', '--ref_dir', required=True, help="Reference audio folder.")
    parser.add_argument('-d', '--est_dir', required=True, help="Estimated audio folder.")
    parser.add_argument('-m', '--metrics', nargs='+', choices=['pesq-nb', 'pesq-wb', 'stoi', 'speakersim', 'melspec-loss'], default=['pesq-nb'], help="List of metrics to evaluate.")
    parser.add_argument('-mp', '--model_path', help="Path to the pre-trained model for speaker similarity.", default="/path/to/wavlm_model.pth")

    args = parser.parse_args()

    results = evaluate(
        ref_folder=args.ref_dir,
        est_folder=args.est_dir,
        metrics=args.metrics,
        model_path=args.model_path
    )

    for metric, data in results.items():
        print(f"{metric.capitalize()}: Mean = {data['mean']}")
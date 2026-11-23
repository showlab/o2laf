# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""Fixed-length video clips from WebDataset tar shards.

Captions are not read: latent action learning is caption-free, and both models
are conditioned on one constant prompt chosen by the trainer.
"""

import math

import torch
from torchvision.transforms import Compose, Normalize, Resize

from o2laf.utils.data_util import retry_load_error
from o2laf.utils.registry import DATASET_REGISTRY
from o2laf.utils.video_util import VideoLoader

from .web_dataset.wids import WebDataset


@DATASET_REGISTRY.register()
class VideoTarDataset(WebDataset):
    """``num_frames`` RGB frames at ``fps`` per sample, resized and scaled to ``[-1, 1]``.

    ``preprocess_video_cfg`` keys: ``height``, ``width``, ``fps``, ``num_frames``,
    ``random_sample`` (random temporal crop) and ``skip_head_seconds``.
    """

    def __init__(self, meta_path, preprocess_video_cfg):
        super().__init__(data_dir=None, meta_path=meta_path)
        self.preprocess_video_cfg = dict(preprocess_video_cfg)
        self.num_frames = int(self.preprocess_video_cfg['num_frames'])
        self.fps = self.preprocess_video_cfg['fps']
        self.random_sample = bool(self.preprocess_video_cfg.get('random_sample', False))
        self.skip_head_seconds = float(self.preprocess_video_cfg.get('skip_head_seconds', 0.0))
        self.transform = Compose([
            Resize((self.preprocess_video_cfg['height'], self.preprocess_video_cfg['width'])),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

    def _head_skip_frames(self, video_loader, skip_seconds, target_fps):
        """Native frames to drop from the clip head, clamped to what the clip affords.

        MiraData segments often open on a fade-in or black frames, whose
        transition into content looks like a large motion the action code would
        be asked to explain. Short clips get a smaller (possibly zero) skip
        rather than an error.
        """
        native_fps = video_loader.get_fps()
        total_frames = video_loader.get_frame_count()
        if not native_fps or total_frames <= 0:
            return 0
        needed = math.ceil(self.num_frames * native_fps / float(target_fps))
        latest_start = total_frames - needed
        if latest_start <= 0:
            return 0
        return int(min(round(skip_seconds * native_fps), latest_start))

    # Web-video corpora contain clips that are too short or malformed; retry
    # with distinct random indices, with a finite bound for a broken dataset.
    @retry_load_error(max_attempts=16)
    def __getitem__(self, index: int):
        sample = self.dataset[index]
        with VideoLoader(sample['.mp4'], name=sample['__key__']) as video_loader:
            source_frame_start = (
                self._head_skip_frames(video_loader, self.skip_head_seconds, self.fps)
                if self.skip_head_seconds > 0
                else 0
            )
            video = video_loader.get_clip(
                self.num_frames,
                target_fps=self.fps,
                random_sample=self.random_sample,
                source_frame_start=source_frame_start,
            )
        if video.shape[0] != self.num_frames:
            raise RuntimeError(f'expected {self.num_frames} frames, got {video.shape[0]}')
        video = torch.from_numpy(video / 255.0).float().permute(0, 3, 1, 2).contiguous()
        return {'pixel_values': self.transform(video), 'fps': self.fps}

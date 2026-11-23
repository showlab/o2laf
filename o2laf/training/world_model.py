"""World model training: a frozen action encoder drives a bidirectional, fully fine-tuned Wan world model.

The action encoder comes from a latent-action run -- its ``latent_action_state.pt``
or a released model directory -- and is frozen, so its action space is fixed. The world model is trained on 81-frame clips with full
attention. The Wan DiT encoder reads the whole 21-latent clip; RGB encoders read
overlapping 41-frame windows that share one frame, which tile all 80
transitions of an 81-frame clip.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from diffusers.training_utils import compute_density_for_timestep_sampling
from einops import repeat

from o2laf.inference import hub
from o2laf.training.base import WanTrainerBase
from o2laf.training.frozen_encoder import (
    FROZEN_ENCODER_ID_BUFFER,
    assert_frozen_encoder,
    id_bytes,
    read_frozen_encoder_source,
)
from o2laf.training.latent_action import LatentActionTrainer
from o2laf.utils.action_util import NULL_ACTION_NO_BIAS
from o2laf.utils.dist_util import load_full_state_dict, raise_if_any_rank_failed, reduce_loss
from o2laf.utils.logger_util import get_logger
from o2laf.utils.lora_util import validate_lora_config
from o2laf.utils.registry import TRAINER_REGISTRY

WORLD_MODEL_INITS = {'pretrained_wan', 'merged_e2e_decoder'}
# Keys of the old file-hash binding; refused so a stale config fails loudly.
RETIRED_FIXED_IDM_KEYS = {
    'state_sha256', 'state_size', 'verify_state_sha256', 'source_global_step', 'source_num_rgb_frames',
    'source_num_latent_frames', 'local_cache_dir', 'init_fdm_from_source_decoder',
}


@TRAINER_REGISTRY.register()
class WorldModelTrainer(LatentActionTrainer):
    """Freeze a trained action encoder and fully fine-tune a bidirectional world model."""

    config_name = 'world_model_trainer.json'
    world_model_causal = False
    # This trainer's dropout drops a whole sample's action bias, so the model
    # learns "no action" as no bias at all -- which is also what the paper's
    # evaluation passes (``actions=None``).
    null_action = NULL_ACTION_NO_BIAS
    disable_lightweight_checkpoint = True
    lightweight_checkpoint_is_distributed = False

    def __init__(
        self,
        transformer_cfg=None,
        idm_cfg=None,
        text_encoder_cfg=None,
        vae_cfg=None,
        scheduler_cfg=None,
        action_cfg=None,
        fixed_idm_cfg=None,
        dtype=torch.float,
        device='cpu',
    ) -> None:
        transformer_cfg = dict(transformer_cfg or {})
        idm_cfg = dict(idm_cfg or {})
        action_cfg = dict(action_cfg or {})
        fixed_idm_cfg = dict(fixed_idm_cfg or {})

        missing = {'state_path', 'fdm_init'} - set(fixed_idm_cfg)
        if missing:
            raise ValueError(f'fixed_idm_cfg is missing required keys {sorted(missing)}.')
        retired = sorted(set(fixed_idm_cfg) & RETIRED_FIXED_IDM_KEYS)
        if retired:
            raise ValueError(
                f'fixed_idm_cfg keys {retired} are no longer used: the encoder is identified by a hash of its '
                'weights, computed from state_path. Remove them.'
            )

        # Needed before super().__init__, which checks the encoder's geometry.
        self.setup_action_encoder_spec(idm_cfg, action_cfg)
        self.legacy_contract_state_keys = self.encoder_spec.legacy_contract_state_keys

        self._validate_config(transformer_cfg, idm_cfg, action_cfg, fixed_idm_cfg)
        self._encoder_source_path = str(fixed_idm_cfg['state_path'])
        self._encoder_fps_values = self._parse_fps_values(fixed_idm_cfg)
        self._encoder_window = self._parse_encoder_window(fixed_idm_cfg)
        self.world_model_init = str(fixed_idm_cfg['fdm_init'])
        # The world model's identity buffer is registered while it is built,
        # inside super().__init__, so the encoder is read and hashed first.
        self._read_frozen_encoder(device, action_cfg)

        # The parent loads the encoder half of the source at the end of its constructor.
        action_cfg['latent_action_state_path'] = self._encoder_source_path
        action_cfg['latent_action_state_parts'] = ['encoder']
        if self.world_model_init == 'merged_e2e_decoder':
            transformer_cfg['merge_init_cfg'] = {'path': self._encoder_source_path}

        super().__init__(
            transformer_cfg=transformer_cfg,
            idm_cfg=idm_cfg,
            text_encoder_cfg=text_encoder_cfg,
            vae_cfg=vae_cfg,
            scheduler_cfg=scheduler_cfg,
            action_cfg=action_cfg,
            dtype=dtype,
            device=device,
        )

        if self.num_rgb_frames != 81 or self.num_latent_frames != 21:
            raise ValueError(
                f'World model training uses 81 RGB / 21 latent frames, got {self.num_rgb_frames} and {self.num_latent_frames}.'
            )
        if self.num_context_frames != 1:
            raise ValueError('World model training uses exactly one clean context frame.')
        if self._encoder_window is not None:
            self.encoder_window_starts(self.num_rgb_frames)

        self.action_encoder.requires_grad_(False)
        self.action_encoder.eval()
        trainable_encoder = sum(p.numel() for p in self.action_encoder.parameters() if p.requires_grad)
        if trainable_encoder:
            raise RuntimeError(f'The frozen action encoder has {trainable_encoder} trainable parameters.')
        trainable = sum(p.numel() for p in self.world_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.world_model.parameters())
        if trainable != total:
            raise RuntimeError(f'Full fine-tuning expected: only {trainable}/{total} world model parameters are trainable.')
        self._assert_frozen_encoder_contract()
        self._action_drop_frac = 0.0
        self._cached_eval_actions = None
        get_logger().info(
            'World model training: encoder=%s (%s, step %d), world model %.3fB/%.3fB trainable, init=%s, '
            'encoder window=%s, source=%s.',
            self.encoder_name, self.frozen_action_space_id, self._encoder_source_global_step,
            trainable / 1e9, total / 1e9, self.world_model_init, self._encoder_window or 'full clip',
            self._encoder_source_path,
        )

    def _read_frozen_encoder(self, device, action_cfg) -> None:
        """Read the source on rank 0, check it, and share what every rank needs to know."""
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self._frozen_source = None
        shared, error = [None], None
        if rank == 0:
            try:
                source = read_frozen_encoder_source(self._encoder_source_path)
                self._check_frozen_encoder(source.metadata, action_cfg)
                self._frozen_source = source
                shared[0] = source.metadata
            except Exception as exception:  # noqa: BLE001 - synchronized below
                error = f'{type(exception).__name__}: {exception}'
        raise_if_any_rank_failed(error, device, 'Frozen encoder validation failed')
        if dist.is_available() and dist.is_initialized():
            dist.broadcast_object_list(shared, src=0)
        metadata = shared[0]
        self.frozen_action_space_id = metadata['action_space_id']
        self._encoder_source_global_step = int(metadata['global_step'])
        self._frozen_encoder_metadata = metadata

    def _check_frozen_encoder(self, metadata, action_cfg) -> None:
        if metadata['encoder_name'] not in (None, self.encoder_spec.name):
            raise RuntimeError(f'{self._encoder_source_path} holds a {metadata["encoder_name"]!r} action encoder, '
                               f'not {self.encoder_spec.name!r}.')
        latent_dim = int(action_cfg.get('latent_action_dim', 32))
        if metadata['latent_dim'] != latent_dim:
            raise RuntimeError(f'The frozen encoder emits {metadata["latent_dim"]}-dimensional codes; the config says {latent_dim}.')
        # The validated recipe: encoders trained on 41-frame (11-latent) clips.
        if (metadata['num_rgb_frames'], metadata['num_latent_frames']) != (41, 11):
            raise RuntimeError(
                f'The frozen encoder was trained on {metadata["num_rgb_frames"]} RGB / {metadata["num_latent_frames"]} '
                'latent frames; world model training expects one from a 41 / 11-frame latent-action run.'
            )

    def _validate_config(self, transformer_cfg, idm_cfg, action_cfg, fixed_idm_cfg) -> None:
        if transformer_cfg.get('lora_config') is not None:
            raise ValueError('World model training fully fine-tunes the world model; remove transformer_cfg.lora_config.')
        if list(transformer_cfg.get('param_names_to_optimize', ())) != ['']:
            raise ValueError('Full fine-tuning requires transformer_cfg.param_names_to_optimize: [""].')
        if transformer_cfg.get('param_names_to_exclude'):
            raise ValueError('Full fine-tuning does not allow transformer_cfg.param_names_to_exclude.')
        if int(action_cfg.get('num_rgb_frames', 0)) != 81:
            raise ValueError('World model training requires action_cfg.num_rgb_frames: 81.')
        if int(action_cfg.get('temporal_compression_ratio', 0)) != 4:
            raise ValueError('World model training requires action_cfg.temporal_compression_ratio: 4.')
        if int(action_cfg.get('num_context_frames', 1)) != 1:
            raise ValueError('World model training requires action_cfg.num_context_frames: 1.')
        if idm_cfg.get('enable_gradient_checkpoint', False):
            raise ValueError('The frozen action encoder runs under no_grad; set idm_cfg.enable_gradient_checkpoint: false.')
        if str(fixed_idm_cfg['fdm_init']) not in WORLD_MODEL_INITS:
            raise ValueError(
                f'fixed_idm_cfg.fdm_init must be one of {sorted(WORLD_MODEL_INITS)}, got {fixed_idm_cfg["fdm_init"]!r}.'
            )

    @staticmethod
    def _parse_fps_values(fixed_idm_cfg) -> tuple[int, ...]:
        values = fixed_idm_cfg.get('fps_values')
        if values is not None and 'fps' in fixed_idm_cfg:
            raise ValueError('fixed_idm_cfg must set either fps or fps_values, not both.')
        if values is None:
            values = (fixed_idm_cfg.get('fps', 16),)
        elif not isinstance(values, (list, tuple)) or not values:
            raise ValueError('fixed_idm_cfg.fps_values must be a non-empty list or tuple.')
        parsed = tuple(sorted({float(value) for value in values}))
        if any(value <= 0 or not value.is_integer() for value in parsed):
            raise ValueError(f'Frozen encoder FPS values must be positive integers, got {parsed}.')
        return tuple(int(value) for value in parsed)

    def _parse_encoder_window(self, fixed_idm_cfg) -> int | None:
        configured = fixed_idm_cfg.get('idm_window_rgb_frames')
        if not self.encoder_spec.reads_pixels:
            if configured is not None:
                raise ValueError(
                    f'The {self.encoder_spec.name} action encoder reads latents, not RGB windows; omit idm_window_rgb_frames.'
                )
            return None
        window = configured if configured is not None else self.encoder_spec.window_rgb_frames
        if window is None:
            return None  # this pixel encoder takes the whole clip
        window = int(window)
        if window < 2 or (window - 1) % 4:
            raise ValueError(f'idm_window_rgb_frames must be 4k+1 so every window ends on a latent boundary, got {window}.')
        return window

    # -- construction and source-state loading ----------------------------

    def build_action_encoder(self, idm_cfg, action_cfg, dtype):
        del action_cfg
        self._idm_contract = self.encoder_spec.build_contract(self, idm_cfg)
        return self.encoder_spec.build_encoder(self, idm_cfg, dtype)

    def _register_world_model_buffers(self, world_model) -> None:
        """Record in the world model which encoder its actions come from.

        A persistent buffer, so every checkpoint of this world model carries it
        and can never be paired with a different action space unnoticed.
        """
        world_model.register_buffer(FROZEN_ENCODER_ID_BUFFER, id_bytes(self.frozen_action_space_id), persistent=True)

    def load_latent_action_weights(self, path):
        """Load the frozen encoder read and checked in the constructor."""
        del path  # already read, on rank 0, by _read_frozen_encoder
        source = self._frozen_source
        state = {}
        if source is not None:
            stored = {hub.to_hub_key(name): tensor for name, tensor in source.encoder.items()}
            names = [name for name, _ in self.action_encoder.named_parameters()]
            names += [name for name, _ in self.action_encoder.named_buffers()]
            mapping = hub.hub_keys_to_model_keys(stored, names)
            state = {mapping[name]: tensor for name, tensor in stored.items()}
            self.encoder_spec.validate_state_contract(
                source.metadata['contract_state'], self._idm_contract, source.path
            )
            validate_lora_config(
                self.action_encoder_lora_config, source.metadata['encoder_lora_config'], 'frozen action encoder'
            )
        load_full_state_dict(
            self.action_encoder,
            state,
            required={name for name, parameter in self.action_encoder.named_parameters() if parameter.requires_grad},
            label='frozen action encoder',
            source_name=self._encoder_source_path,
            device=self.config.device,
            unexpected='raise',
        )
        get_logger().info(
            'Loaded frozen %s action encoder %s from %s (step %d).',
            self.encoder_name, self.frozen_action_space_id, self._encoder_source_path, self._encoder_source_global_step,
        )
        self._frozen_source = None

    def _assert_frozen_encoder_contract(self) -> None:
        assert_frozen_encoder(self.world_model, self.frozen_action_space_id)

    def set_ema_model(self, decay, warmup_step=0, decay_type='fixed'):
        """EMA is coherent here: only the world model is optimised."""
        return WanTrainerBase.set_ema_model(self, decay, warmup_step=warmup_step, decay_type=decay_type)

    # -- frozen action extraction -----------------------------------------

    def encoder_window_starts(self, num_rgb_frames: int) -> list[int]:
        """Starts of overlapping RGB windows that share one frame."""
        if self._encoder_window is None or num_rgb_frames <= self._encoder_window:
            return [0]
        stride = self._encoder_window - 1
        if (num_rgb_frames - 1) % stride:
            raise ValueError(f'{num_rgb_frames} RGB frames cannot be tiled by {self._encoder_window}-frame windows.')
        return list(range(0, num_rgb_frames - 1, stride))

    @torch.no_grad()
    def windowed_actions(self, latents, prompt_embeds, pixels=None) -> dict[str, torch.Tensor]:
        """Encode every window of the clip in one batch and stitch the codes in time order.

        An encoder that takes the whole clip -- the Wan DiT encoder, whose
        preprocessing is the VAE the caller already ran -- is the one-window
        case, so it needs no separate path.
        """
        self.action_encoder.eval()
        batch = latents.shape[0] if latents is not None else pixels.shape[0]
        starts = [0] if pixels is None else self.encoder_window_starts(pixels.shape[1])
        windows = pixels
        if len(starts) > 1:
            windows = torch.cat([pixels[:, start:start + self._encoder_window] for start in starts], dim=0)
        outputs = self.encoder_spec.encode(
            self.action_encoder, latents=latents, pixel_values=windows, prompt_embeds=prompt_embeds
        )
        window_frames = windows.shape[1] if windows is not None else self.num_rgb_frames
        expected_per_window = self.action_compression_ratio * ((window_frames - 1) // self.temporal_compression_ratio)
        stitched = {}
        for key in ('action_mu', 'action_logvar'):
            value = outputs[key]
            expected = (batch * len(starts), expected_per_window, self.latent_action_dim)
            if tuple(value.shape) != expected:
                raise RuntimeError(f'Frozen {self.encoder_name} encoder {key} shape {tuple(value.shape)}, expected {expected}.')
            stitched[key] = (
                value.reshape(len(starts), batch, expected_per_window, self.latent_action_dim)
                .transpose(0, 1)
                .reshape(batch, -1, self.latent_action_dim)
            )
        stitched['action'] = stitched['action_mu']
        return stitched

    @torch.no_grad()
    def encode_actions(self, latents, prompt_embeds, batch=None):
        """Posterior means of the frozen encoder, over as many windows as the clip needs."""
        pixels = None
        if batch is not None and 'pixel_values' in batch:
            pixels = batch['pixel_values'].to(device=self.config.device, dtype=self.config.dtype)
        elif self.encoder_spec.reads_pixels:
            raise ValueError(f'The {self.encoder_name} action encoder reads pixels and requires batch["pixel_values"].')
        outputs = self.windowed_actions(latents, prompt_embeds, pixels=pixels)
        if latents is not None:
            # The code conditions these latents, so it is carried in their dtype.
            outputs['action'] = outputs['action'].to(dtype=latents.dtype)
        return outputs

    # -- objective --------------------------------------------------------

    def _action_keep_mask(self, batch_size: int, device):
        """Per-sample action dropout: a dropped sample sees no action bias at all."""
        if self.action_dropout_prob <= 0:
            self._action_drop_frac = 0.0
            return None
        keep = torch.rand(batch_size, device=device) >= self.action_dropout_prob
        self._action_drop_frac = 1.0 - float(keep.float().mean())
        return keep

    def flow_matching_loss(self, latents, packed_actions, prompt_embeds, *, noise=None, raw_timestep=None, context_length=None):
        """Unweighted flow-matching loss with logit-normal timesteps.

        Evaluation passes ``noise`` and ``raw_timestep`` so every action variant
        shares one diffusion condition; it then also disables action dropout.
        """
        batch_size, _, num_frames, _, _ = latents.shape
        if num_frames != self.num_latent_frames:
            raise ValueError(f'Wan VAE produced {num_frames} latent frames, expected {self.num_latent_frames}.')
        deterministic_eval = raw_timestep is not None
        if raw_timestep is None:
            raw_timestep = compute_density_for_timestep_sampling(
                weighting_scheme='logit_normal', batch_size=batch_size, logit_mean=0.0, logit_std=1.0
            ).to(device=latents.device, dtype=torch.float32)
        else:
            raw_timestep = raw_timestep.to(device=latents.device, dtype=torch.float32).reshape(batch_size)
        timestep = self.scheduler.apply_shift(raw_timestep)
        timestep = repeat(timestep, 'b -> b f', b=batch_size, f=num_frames).contiguous()
        timestep = timestep * self.scheduler.config.num_train_timesteps
        if noise is None:
            noise = torch.randn_like(latents)
        elif noise.shape != latents.shape:
            raise ValueError(f'noise shape {tuple(noise.shape)} does not match latents {tuple(latents.shape)}.')
        else:
            noise = noise.to(device=latents.device, dtype=latents.dtype)
        noisy = self.scheduler.scale_noise(latents, timestep[:, None, :], noise)

        context_length = self.num_context_frames if context_length is None else int(context_length)
        if not 0 <= context_length < num_frames:
            raise ValueError(f'context_length={context_length} must be in [0, {num_frames}).')
        if context_length:
            noisy = torch.cat([latents[:, :, :context_length], noisy[:, :, context_length:]], dim=2)
            timestep = torch.cat([torch.zeros_like(timestep[:, :context_length]), timestep[:, context_length:]], dim=1)

        prediction = self.world_model(
            noisy,
            chunk_partition=[1] * num_frames,
            timestep=timestep,
            r_timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            actions=packed_actions,
            action_keep_mask=(None if deterministic_eval else self._action_keep_mask(batch_size, latents.device)),
            return_dict=False,
        )
        target = noise - latents
        per_frame = (prediction.float() - target.float()).square().mean(dim=(1, 3, 4))
        return per_frame[:, context_length:].mean()

    def _validate_batch_fps(self, fps) -> None:
        if torch.is_tensor(fps):
            values = fps.detach().cpu().flatten().tolist()
        elif isinstance(fps, (list, tuple)):
            values = list(fps)
        else:
            values = [fps]
        observed = tuple(sorted({float(value) for value in values}))
        invalid = tuple(
            value for value in observed
            if not any(abs(value - allowed) <= 1e-6 for allowed in self._encoder_fps_values)
        )
        if invalid:
            raise ValueError(f'Batch FPS values {invalid} are outside the frozen encoder FPS set {self._encoder_fps_values}.')

    def train_step(self, batch, iters=-1):
        del iters
        self.world_model.train()
        self.action_encoder.eval()
        self._assert_frozen_encoder_contract()

        pixel_values = batch['pixel_values'].to(device=self.config.device)
        if pixel_values.dtype == torch.uint8:
            pixel_values = pixel_values.to(dtype=self.config.dtype).div_(127.5).sub_(1.0)
        else:
            pixel_values = pixel_values.to(dtype=self.config.dtype)
        latents = self.encode_latents(pixel_values)
        if latents.shape[2] != self.num_latent_frames:
            raise ValueError(f'Batch has {latents.shape[2]} latent frames, expected {self.num_latent_frames}.')
        if 'fps' in batch:
            self._validate_batch_fps(batch['fps'])

        prompt_embeds = self.fixed_prompt_embedding(latents.shape[0], latents.device)
        action_batch = dict(batch)
        action_batch['pixel_values'] = pixel_values
        action_outputs = self.encode_actions(latents, prompt_embeds, batch=action_batch)
        actions = action_outputs['action'].to(dtype=latents.dtype)
        expected = (latents.shape[0], self.expected_num_actions, self.latent_action_dim)
        if tuple(actions.shape) != expected:
            raise RuntimeError(f'Frozen {self.encoder_name} encoder returned {tuple(actions.shape)}, expected {expected}.')
        if not bool(torch.isfinite(actions).all()):
            raise FloatingPointError('The frozen action encoder returned NaN or Inf actions.')

        flow_loss = self.flow_matching_loss(latents, self.pack_actions(actions), prompt_embeds)
        flow_loss.backward()

        action_mu = action_outputs['action_mu'].float()
        action_logvar = action_outputs['action_logvar'].float()
        logs = {
            'flow_loss': float(reduce_loss(flow_loss)),
            'action_mean': float(action_mu.mean()),
            'action_std': float(action_mu.std()),
            'action_abs_mean': float(action_mu.abs().mean()),
            'action_sigma': float(torch.exp(0.5 * action_logvar).mean()),
            'action_drop_frac': self._action_drop_frac,
        }
        embedder = getattr(self.world_model, 'action_embedder', None)
        if embedder is not None:
            gamma = embedder.gamma.detach().float()
            if hasattr(gamma, 'full_tensor'):
                gamma = gamma.full_tensor()
            logs['gamma'] = float(gamma.reshape(-1)[0])
            logs['action_gain'] = embedder.effective_gain()
        return logs

    # -- evaluation --------------------------------------------------------

    def eval_action_codes(self, clips, prompt_embeds, eval_cfg):
        if self._cached_eval_actions is not None:
            return self._cached_eval_actions
        height = int(eval_cfg.get('height', 480))
        width = int(eval_cfg.get('width', 832))
        fps = int(eval_cfg.get('fps', 16))
        num_frames = int(eval_cfg.get('num_frames', self.num_rgb_frames))
        codes = {}
        for entry in eval_cfg['clips']:
            name = str(entry['name'])
            action_batch = None
            if self.encoder_spec.reads_pixels:
                # Only an encoder that reads pixels needs the clip decoded again,
                # at that clip's own frame rate.
                clip_fps = int(entry.get('fps', fps))
                pixels = self.load_eval_clip(entry['path'], height, width, clip_fps, num_frames=num_frames).to(
                    device=self.config.device, dtype=self.config.dtype
                )
                action_batch = {'pixel_values': pixels}
            codes[name] = self.encode_actions(clips.get(name), prompt_embeds, batch=action_batch)['action'].to(
                device=self.config.device, dtype=self.config.dtype
            )
        self._cached_eval_actions = codes
        return self._cached_eval_actions

    def _eval_action_encoder_restore(self):
        self.action_encoder.eval()

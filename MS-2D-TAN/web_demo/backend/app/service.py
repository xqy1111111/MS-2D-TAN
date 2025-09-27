from __future__ import annotations

import copy
from dataclasses import dataclass
from importlib import reload
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch

# Ensure lib/ is on sys.path before importing project modules
import moment_localization._init_paths  # noqa: F401
from lib.datasets.dataset import MomentLocalizationDataset
from lib.core import config as config_module
import models  # noqa: F401

from .config import DATASETS, DatasetConfig, DEFAULT_DEVICE, DEFAULT_DATA_ROOT


@dataclass
class VideoMeta:
    duration: float
    examples: List[dict]


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state_dict.items()}


def _load_cfg(cfg_path: Path):
    module = reload(config_module)
    module.update_config(str(cfg_path))
    return copy.deepcopy(module.cfg)


class DatasetRuntime:
    def __init__(
        self,
        dataset_key: str,
        cfg_definition: DatasetConfig,
        data_root: Path,
        device: torch.device,
    ) -> None:
        self.dataset_key = dataset_key
        self.definition = cfg_definition
        self.device = device
        self.cfg = _load_cfg(cfg_definition.cfg_path)

        configured_data_dir = Path(self.cfg.DATASET.DATA_DIR)
        if configured_data_dir.is_absolute():
            data_dir = configured_data_dir
        else:
            repo_root = cfg_definition.cfg_path.parents[2]
            if configured_data_dir.parts and configured_data_dir.parts[0] == "data":
                relative = Path(*configured_data_dir.parts[1:]) if len(configured_data_dir.parts) > 1 else Path()
                data_dir = (data_root / relative).resolve()
            else:
                data_dir = (repo_root / configured_data_dir).resolve()
        self.cfg.DATASET.DATA_DIR = str(data_dir)
        self.cfg.DATASET.SPLIT = cfg_definition.split

        self.model = self._load_model(cfg_definition.checkpoint_path)
        self.dataset = MomentLocalizationDataset(self.cfg.DATASET, cfg_definition.split)
        self.video_meta: Dict[str, VideoMeta] = {}
        for anno in self.dataset.annotations:
            video_id = anno['video']
            entry = self.video_meta.setdefault(video_id, VideoMeta(duration=anno['duration'], examples=[]))
            if len(entry.examples) < 5:
                entry.examples.append({
                    'description': anno['description'],
                    'times': anno['times'],
                })
        self.sorted_video_ids = sorted(self.video_meta.keys())
        self.time_unit = None
        if getattr(self.cfg.DATASET, 'SLIDING_WINDOW', False):
            output_clips = self.cfg.DATASET.OUTPUT_NUM_CLIPS
            if isinstance(output_clips, list):
                output_clips = output_clips[0]
            self.time_unit = self.cfg.DATASET.TIME_UNIT * self.cfg.DATASET.INPUT_NUM_CLIPS / output_clips
        self.nms_threshold = self.cfg.TEST.NMS_THRESH
        self.max_top_k = self.cfg.TEST.TOP_K
        self._lock = threading.Lock()

    def _load_model(self, checkpoint_path: Path) -> torch.nn.Module:
        checkpoint_path = checkpoint_path if checkpoint_path.is_absolute() else checkpoint_path.resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at {checkpoint_path}. Download the release weights and place them accordingly."
            )
        model_class = getattr(models, self.cfg.MODEL.NAME)
        model = model_class(self.cfg.MODEL)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, dict):
            for key in ('state_dict', 'model', 'module'):
                if key in checkpoint:
                    checkpoint = checkpoint[key]
                    break
        state_dict = _strip_module_prefix(checkpoint)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def list_videos(self, limit: int = 20, offset: int = 0) -> Tuple[List[str], int]:
        total = len(self.sorted_video_ids)
        end = min(offset + limit, total)
        return self.sorted_video_ids[offset:end], total

    def get_examples(self, video_id: str) -> VideoMeta:
        if video_id not in self.video_meta:
            raise KeyError(f"Unknown video id '{video_id}' for dataset '{self.dataset_key}'.")
        return self.video_meta[video_id]

    def localize(self, video_id: str, description: str, top_k: int, nms_threshold: Optional[float] = None) -> Dict[str, object]:
        if video_id not in self.video_meta:
            raise KeyError(f"Video '{video_id}' is not indexed for dataset '{self.dataset_key}'.")
        top_k = min(top_k, self.max_top_k)
        proposal_threshold = nms_threshold if nms_threshold is not None else self.nms_threshold

        with self._lock:
            word_vectors, txt_mask = self.dataset.get_sentence_features(description)
            video_features, vis_mask = self.dataset.get_video_features(video_id)

        textual_input = word_vectors.unsqueeze(0).float().to(self.device)
        textual_mask = txt_mask.unsqueeze(0).float().to(self.device)
        visual_input = video_features.unsqueeze(0).transpose(1, 2).float().to(self.device)
        visual_mask = vis_mask.unsqueeze(0).transpose(1, 2).float().to(self.device)

        with torch.no_grad():
            predictions, map_masks = self.model(textual_input, textual_mask, visual_input, visual_mask)
        score_map = self._recover_to_single_map(predictions)
        mask_map = self._recover_to_single_map(map_masks)
        if torch.sum(mask_map[0] > 0).item() == 0:
            raise RuntimeError("No valid proposals produced for the given sample.")

        if getattr(self.cfg.DATASET, 'SLIDING_WINDOW', False):
            duration = self.video_meta[video_id].duration
            stride = self.time_unit or self.cfg.DATASET.TIME_UNIT
            proposals = self._decode_sliding_window(score_map, mask_map, stride, duration, top_k, proposal_threshold)
            feature_stride = stride
        else:
            duration = self.video_meta[video_id].duration
            proposals = self._decode_fixed_window(score_map, mask_map, duration, top_k, proposal_threshold)
            feature_stride = duration / self.cfg.DATASET.INPUT_NUM_CLIPS

        return {
            'duration': duration,
            'segments': proposals,
            'feature_length': int(video_features.shape[0]),
            'feature_stride': feature_stride,
        }

    def _recover_to_single_map(self, maps: Iterable[torch.Tensor]) -> torch.Tensor:
        maps = [m.detach().to(self.device) for m in maps]
        batch_size, _, map_size, _ = maps[0].shape
        score_map = torch.zeros(batch_size, 1, map_size, map_size, device=self.device)
        for prob in maps:
            scale_num_clips, scale_num_anchors = prob.shape[2:]
            dilation = map_size // scale_num_clips
            upto = map_size // dilation * dilation
            for anchor_idx in range(scale_num_anchors):
                current = prob[..., anchor_idx]
                target_slice = score_map[..., :upto:dilation, (anchor_idx + 1) * dilation - 1]
                updated = torch.max(target_slice, current)
                score_map[..., :upto:dilation, (anchor_idx + 1) * dilation - 1] = updated
        return score_map.cpu()

    def _decode_sliding_window(
        self,
        scores: torch.Tensor,
        mask: torch.Tensor,
        time_unit: float,
        duration: float,
        top_k: int,
        nms_thresh: float,
    ) -> List[Dict[str, float]]:
        segments = self._gather_segments(scores, mask, time_unit)
        return self._nms_segments(segments, top_k, nms_thresh, duration)

    def _decode_fixed_window(
        self,
        scores: torch.Tensor,
        mask: torch.Tensor,
        duration: float,
        top_k: int,
        nms_thresh: float,
    ) -> List[Dict[str, float]]:
        time_unit = duration / mask.shape[-2]
        segments = self._gather_segments(scores, mask, time_unit)
        return self._nms_segments(segments, top_k, nms_thresh, duration)

    def _gather_segments(
        self,
        scores: torch.Tensor,
        mask: torch.Tensor,
        time_unit: float,
    ) -> List[Tuple[float, float, float]]:
        batch_size, _, num_clips, num_anchors = scores.shape
        valid_k = int(torch.sum(mask[0] > 0).item())
        scores_flat = scores.view(batch_size, -1)
        ranking_scores, indices = torch.topk(scores_flat, valid_k, dim=1)

        segments: List[Tuple[float, float, float]] = []
        for score_value, index_value in zip(ranking_scores[0], indices[0]):
            clip_index = (index_value // num_anchors).float()
            anchor_index = (index_value % num_anchors).float()
            start = clip_index.item() * time_unit
            end = start + (anchor_index.item() + 1) * time_unit
            segments.append((max(0.0, start), end, score_value.item()))
        return segments

    def _nms_segments(
        self,
        segments: List[Tuple[float, float, float]],
        top_k: int,
        threshold: float,
        duration: float,
    ) -> List[Dict[str, float]]:
        if not segments:
            return []
        segments = sorted(segments, key=lambda x: x[2], reverse=True)
        selected: List[Tuple[float, float, float]] = []
        while segments and len(selected) < top_k:
            current = segments.pop(0)
            selected.append(current)
            segments = [seg for seg in segments if self._iou_1d(current, seg) <= threshold]

        formatted = []
        for start, end, score in selected:
            formatted.append({
                'start': float(max(0.0, start)),
                'end': float(min(duration, end)),
                'score': float(score),
            })
        return formatted

    @staticmethod
    def _iou_1d(seg_a: Tuple[float, float, float], seg_b: Tuple[float, float, float]) -> float:
        start = max(seg_a[0], seg_b[0])
        end = min(seg_a[1], seg_b[1])
        if end <= start:
            return 0.0
        inter = end - start
        union = (seg_a[1] - seg_a[0]) + (seg_b[1] - seg_b[0]) - inter
        return inter / union if union > 0 else 0.0


class MS2DTANService:
    def __init__(
        self,
        dataset_configs: Dict[str, DatasetConfig] = DATASETS,
        data_root: Optional[Path] = None,
        device_hint: Optional[str] = None,
    ) -> None:
        self.dataset_configs = dataset_configs
        base_data_root = Path(data_root or DEFAULT_DATA_ROOT).resolve()
        self.data_root = base_data_root
        desired_device = device_hint or DEFAULT_DEVICE
        if desired_device.startswith("cuda") and not torch.cuda.is_available():
            desired_device = "cpu"
        self.device = torch.device(desired_device)
        self._runtimes: Dict[str, DatasetRuntime] = {}
        self._lock = threading.Lock()

    def get_dataset_metadata(self) -> List[Dict[str, object]]:
        return [cfg.to_metadata() for cfg in self.dataset_configs.values()]

    def _get_runtime(self, dataset_key: str) -> DatasetRuntime:
        if dataset_key not in self.dataset_configs:
            raise KeyError(f"Dataset '{dataset_key}' is not registered.")
        if dataset_key not in self._runtimes:
            with self._lock:
                if dataset_key not in self._runtimes:
                    definition = self.dataset_configs[dataset_key]
                    runtime = DatasetRuntime(dataset_key, definition, self.data_root, self.device)
                    self._runtimes[dataset_key] = runtime
        return self._runtimes[dataset_key]

    def list_videos(self, dataset_key: str, limit: int = 20, offset: int = 0) -> Tuple[List[str], int]:
        runtime = self._get_runtime(dataset_key)
        return runtime.list_videos(limit=limit, offset=offset)

    def get_examples(self, dataset_key: str, video_id: str) -> VideoMeta:
        runtime = self._get_runtime(dataset_key)
        return runtime.get_examples(video_id)

    def localize(self, dataset_key: str, video_id: str, description: str, top_k: int, nms_threshold: Optional[float] = None) -> Dict[str, object]:
        runtime = self._get_runtime(dataset_key)
        return runtime.localize(video_id, description, top_k, nms_threshold)


__all__ = ["MS2DTANService", "VideoMeta"]

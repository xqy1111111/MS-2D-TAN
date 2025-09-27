from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import os


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = os.getenv("MS2DTAN_DATA_ROOT", str(REPO_ROOT / "data"))
DEFAULT_CHECKPOINT_ROOT = os.getenv("MS2DTAN_CHECKPOINT_ROOT", str(REPO_ROOT / "release_checkpoints"))
DEFAULT_DEVICE = os.getenv("MS2DTAN_DEVICE", "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    display_name: str
    cfg_path: Path
    checkpoint_path: Path
    split: str
    description: str
    feature_requirements: str
    best_metrics: Dict[str, float] = field(default_factory=dict)
    default_top_k: int = 5
    notes: Optional[List[str]] = None

    def to_metadata(self) -> Dict[str, object]:
        return {
            "key": self.key,
            "display_name": self.display_name,
            "cfg_path": str(self.cfg_path.relative_to(REPO_ROOT)),
            "checkpoint_path": str(self.checkpoint_path.relative_to(REPO_ROOT)),
            "split": self.split,
            "description": self.description,
            "feature_requirements": self.feature_requirements,
            "best_metrics": self.best_metrics,
            "default_top_k": self.default_top_k,
            "notes": self.notes or [],
        }


def _build_datasets() -> Dict[str, DatasetConfig]:
    checkpoint_root = Path(DEFAULT_CHECKPOINT_ROOT)
    return {
        "charades": DatasetConfig(
            key="charades",
            display_name="Charades-STA – I3D* (finetuned)",
            cfg_path=REPO_ROOT / "experiments/charades/MS-2D-TAN-G-I3D-Finetuned.yaml",
            checkpoint_path=checkpoint_root / "charades/MS-2D-TAN-G-I3D-Finetuned/epoch0081-0.6008-0.8780.pkl",
            split="test",
            description="Highest reported Charades-STA checkpoint using finetuned I3D features (Rank1@0.5 = 0.6008).",
            feature_requirements="Requires Charades-STA RGB/I3D features at 25fps stored under data/Charades-STA/charades_i3d_rgb_25fps.hdf5.",
            best_metrics={
                "Rank1@0.5": 0.6008,
                "Rank1@0.7": 0.3739,
                "Rank5@0.5": 0.8906,
            },
            notes=["Ensure finetuned I3D features are used; rename paths if you keep multiple feature sets."],
        ),
        "activitynet": DatasetConfig(
            key="activitynet",
            display_name="ActivityNet Captions – C3D",
            cfg_path=REPO_ROOT / "experiments/activitynet/MS-2D-TAN-G-C3D.yaml",
            checkpoint_path=checkpoint_root / "activitynet/MS-2D-TAN-G-C3D/epoch0006-0.6104-0.8730.pkl",
            split="test",
            description="Best ActivityNet Captions release (C3D features, Rank1@0.3 = 0.6104, Rank5@0.5 = 0.7880).",
            feature_requirements="Requires ActivityNet Captions C3D features merged into data/ActivityNet/ subfolders as in REPRODUCTION.md.",
            best_metrics={
                "Rank1@0.3": 0.6104,
                "Rank1@0.5": 0.4616,
                "Rank5@0.5": 0.7880,
            },
        ),
        "tacos": DatasetConfig(
            key="tacos",
            display_name="TACoS – VGG",
            cfg_path=REPO_ROOT / "experiments/tacos/MS-2D-TAN-G-VGG.yaml",
            checkpoint_path=checkpoint_root / "tacos/MS-2D-TAN-G-VGG/epoch0034-0.5064-0.7831.pkl",
            split="test",
            description="Best TACoS release using VGG features (Rank1@0.3 = 0.4331, Rank5@0.1 = 0.7831).",
            feature_requirements="Requires TACoS VGG features merged into data/TACoS/tall_c3d_64_features.hdf5.",
            best_metrics={
                "Rank1@0.3": 0.4331,
                "Rank1@0.5": 0.3527,
                "Rank5@0.1": 0.7831,
            },
            notes=["Charades features also work on TACoS, but VGG release offers better Rank1@0.3."],
        ),
    }


DATASETS: Dict[str, DatasetConfig] = _build_datasets()


def resolve_data_root(dataset_key: str, override: Optional[str] = None) -> Path:
    base = Path(override or DEFAULT_DATA_ROOT)
    return base


__all__ = [
    "DatasetConfig",
    "DATASETS",
    "DEFAULT_DATA_ROOT",
    "DEFAULT_CHECKPOINT_ROOT",
    "DEFAULT_DEVICE",
    "REPO_ROOT",
    "resolve_data_root",
]

# src/data/base_dataset.py
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from typing import Dict, Any, List, Callable


class BaseMultimodalDataset(Dataset, ABC):
    """
    Abstract base class for all multimodal datasets.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pass

    @abstractmethod
    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Dataset-specific collate function.
        """
        pass

    @abstractmethod
    def info(self) -> Dict[str, Any]:
        """
        Return dataset metadata for logging/debug.
        """
        pass
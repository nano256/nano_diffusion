import torch
import numpy as np
from streaming import Stream, StreamingDataset
from torch.utils.data import DataLoader
from typing import List, Dict, Union, Optional


class StreamingMockLatentsDataset(StreamingDataset):
    """Dataset class that gives back random latents to test the training script. It mimicks the StreamingLatensDataset

    Args:
        streams: List of individual streams (in our case streams of individual datasets)
        shuffle: Whether to shuffle the dataset
        image_size: Size of images (256 or 512)
        cap_seq_size: Context length of text-encoder
        cap_emb_dim: Dimension of caption embeddings
        cap_drop_prob: Probability of using all zeros caption embedding (classifier-free guidance)
        batch_size: Batch size for streaming
    """

    def __init__(
        self,
        streams: Optional[List[Stream]] = None,
        shuffle: bool = False,
        image_size: Optional[int] = None,
        cap_seq_size: Optional[int] = None,
        cap_emb_dim: Optional[int] = None,
        cap_drop_prob: float = 0.0,
        batch_size: Optional[int] = None,
        num_samples: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            streams=streams,
            shuffle=shuffle,
            batch_size=batch_size,
        )

        self.image_size = image_size
        self.cap_seq_size = cap_seq_size
        self.cap_emb_dim = cap_emb_dim
        self.cap_drop_prob = cap_drop_prob
        # Take twice the batch size if not set
        self.num_samples = (2 * batch_size) if num_samples is None else num_samples

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, str, float]]:
        out = {}

        # Mask for zero'ed out captions in classifier-free guidance (cfg) training.
        # We replace caption embeddings with a zero vector in cfg guidance.
        out["drop_caption_mask"] = 0.0 if torch.rand(1) < self.cap_drop_prob else 1.0
        out["caption_latents"] = torch.rand(
            1, self.cap_seq_size, self.cap_emb_dim, dtype=torch.float16
        )
        out["image_latents"] = torch.rand(
            1, self.image_size, self.image_size, dtype=torch.float16
        )
        return out


def build_streaming_mock_latents_dataloader(
    datadir: Union[str, List[str]],
    batch_size: int,
    image_size: int = 256,
    cap_seq_size: int = 77,
    cap_emb_dim: int = 1024,
    cap_drop_prob: float = 0.0,
    shuffle: bool = True,
    drop_last: bool = True,
    **dataloader_kwargs,
) -> DataLoader:
    """Creates a DataLoader for streaming mock latents."""
    streams = []

    dataset = StreamingMockLatentsDataset(
        streams=streams,
        shuffle=shuffle,
        image_size=image_size,
        cap_seq_size=cap_seq_size,
        cap_emb_dim=cap_emb_dim,
        cap_drop_prob=cap_drop_prob,
        batch_size=batch_size,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=None,
        drop_last=drop_last,
        **dataloader_kwargs,
    )

    return dataloader

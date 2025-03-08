import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Union, Optional


# Don't inherit from StreamingDataset if you don't need its functionality
class MockLatentsDataset(Dataset):
    """Dataset class that gives back random latents to test the training script.

    Args:
        image_size: Size of images (256 or 512)
        cap_seq_size: Context length of text-encoder
        cap_emb_dim: Dimension of caption embeddings
        cap_drop_prob: Probability of using all zeros caption embedding (classifier-free guidance)
        num_samples: Number of samples in the dataset
    """

    def __init__(
        self,
        image_size: Optional[int] = None,
        cap_seq_size: Optional[int] = None,
        cap_emb_dim: Optional[int] = None,
        cap_drop_prob: float = 0.0,
        num_samples: int = 16,
        **kwargs,
    ) -> None:
        self.image_size = image_size
        self.cap_seq_size = cap_seq_size
        self.cap_emb_dim = cap_emb_dim
        self.cap_drop_prob = cap_drop_prob
        self.length = num_samples

    def __len__(self):
        return self.length

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, str, float]]:
        if index >= self.length:
            raise IndexError("list index out of range")

        out = {}
        # Mask for zero'ed out captions in classifier-free guidance (cfg) training.
        # We replace caption embeddings with a zero vector in cfg guidance.
        out["drop_caption_mask"] = 0.0 if torch.rand(1) < self.cap_drop_prob else 1.0
        out["caption_latents"] = torch.rand(
            1, self.cap_seq_size, self.cap_emb_dim, dtype=torch.float16
        )

        # For image latents, make sure dimensions match expectations
        # If image_size is 256, latent_size should be 32 (for 8x downsampling)
        latent_size = self.image_size // 8
        out["image_latents"] = torch.rand(
            4, latent_size, latent_size, dtype=torch.float16  # 4 channels
        )

        # Optional: add caption for debugging
        out["caption"] = f"Mock caption {index}"

        return out


def build_mock_latents_dataloader(
    datadir: Union[str, List[str]],  # Keep for API compatibility but don't use
    batch_size: int,
    image_size: int = 256,
    cap_seq_size: int = 77,
    cap_emb_dim: int = 1024,
    cap_drop_prob: float = 0.0,
    shuffle: bool = True,
    drop_last: bool = True,
    **dataloader_kwargs,
) -> DataLoader:
    """Creates a DataLoader for mock latents."""
    dataset = MockLatentsDataset(
        image_size=image_size,
        cap_seq_size=cap_seq_size,
        cap_emb_dim=cap_emb_dim,
        cap_drop_prob=cap_drop_prob,
        num_samples=batch_size * 2,  # Just to have a few batches
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        **dataloader_kwargs,
    )

    return dataloader

import os
import json
import numpy as np
import torch

# Create directories
os.makedirs("datadir/test_latents/train/0", exist_ok=True)

# Number of samples
num_samples = 100

# Generate random latents
latents_64 = torch.randn(num_samples, 4, 8, 8).half().numpy()  # 4 channels, 8x8 resolution
caption_latents = torch.randn(num_samples, 77, 1024).half().numpy()  # 77 tokens, 1024 dim
captions = ["Random caption " + str(i) for i in range(num_samples)]

# Create MDS file
shard_data = []
for i in range(num_samples):
    sample = {
        "caption": captions[i],
        "caption_latents": caption_latents[i].tobytes(),
        "latents_64": latents_64[i].tobytes()
    }
    shard_data.append(sample)

# Write data to file
with open("datadir/test_latents/train/0/shard.00000.mds", "wb") as f:
    for sample in shard_data:
        # Write caption
        caption_bytes = sample["caption"].encode("utf-8")
        f.write(len(caption_bytes).to_bytes(4, byteorder="little"))
        f.write(caption_bytes)
        
        # Write caption latents
        f.write(len(sample["caption_latents"]).to_bytes(4, byteorder="little"))
        f.write(sample["caption_latents"])
        
        # Write image latents
        f.write(len(sample["latents_64"]).to_bytes(4, byteorder="little"))
        f.write(sample["latents_64"])

# Create index.json
index_data = {
    "shards": [
        {
            "column_encodings": ["str", "bytes", "bytes"],
            "column_names": ["caption", "caption_latents", "latents_64"],
            "column_sizes": [None, None, None],
            "compression": None,
            "format": "mds",
            "hashes": [],
            "raw_data": {
                "basename": "shard.00000.mds",
                "bytes": os.path.getsize("datadir/test_latents/train/0/shard.00000.mds"),
                "hashes": {}
            },
            "samples": num_samples,
            "size_limit": 268435456,
            "version": 2,
            "zip_data": None
        }
    ],
    "version": 2
}

# Write index.json
with open("datadir/test_latents/train/0/index.json", "w") as f:
    json.dump(index_data, f)

# Copy index.json to parent directory
with open("datadir/test_latents/train/index.json", "w") as f:
    json.dump(index_data, f)

print(f"Created test latents with {num_samples} samples") 
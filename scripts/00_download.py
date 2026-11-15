"""Download Flickr8k (HF mirror `jxie/flickr8k`) and the NLTK resources used for claim extraction.

    python scripts/00_download.py
"""
import nltk
from huggingface_hub import snapshot_download

import _common  # noqa: F401
from lcr import config

snapshot_download(repo_id="jxie/flickr8k", repo_type="dataset",
                  local_dir=str(config.DATA / "flickr8k"), allow_patterns=["data/*.parquet", "README.md"])
for res in ("averaged_perceptron_tagger_eng", "punkt_tab", "wordnet"):
    nltk.download(res, download_dir=str(config.DATA / "nltk"), quiet=True)
print("flickr8k + nltk ready under", config.DATA)

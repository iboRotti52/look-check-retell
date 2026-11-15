#!/usr/bin/env bash
# Download data/cache (OWLv2 proposals, BLIP features, decoder_A checkpoint, Look maps, pipeline records) from the
# GitHub release `data-v1` and reassemble it. Run from the repository root:  bash scripts/fetch_data.sh
# Raw images are NOT in the release: run `python scripts/00_download.py` for Flickr8k + NLTK data.
set -euo pipefail
REPO=iboRotti52/look-check-retell; TAG=data-v1; DL=data/_release
mkdir -p "$DL" data/cache/blip
if command -v gh >/dev/null 2>&1; then
  gh release download "$TAG" -R "$REPO" -D "$DL" --skip-existing
else
  urls=$(curl -fsSL "https://api.github.com/repos/$REPO/releases/tags/$TAG" |
         python3 -c "import json,sys; print('\n'.join(a['browser_download_url'] for a in json.load(sys.stdin)['assets']))")
  for u in $urls; do curl -fL -C - -o "$DL/$(basename "$u")" "$u"; done
fi
(cd "$DL" && shasum -a 256 -c SHA256SUMS)
for t in "$DL"/cache_*.tar; do tar -xf "$t" -C data; done
cat "$DL"/cache_blip_train.f16.npy.part-* > data/cache/blip/train.f16.npy
cp "$DL"/cache_blip_validation.f16.npy data/cache/blip/validation.f16.npy
echo "data/cache ready ($(du -sh data/cache | cut -f1)). You may delete $DL."

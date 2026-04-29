#!/bin/bash
# Upload model weights and ChromaDB to Modal volumes
# Run this before deploying

set -e

echo "=== Uploading Labkovsky model to Modal ==="

# Upload AWQ model weights
echo "Uploading model weights..."
modal volume put labkovsky-model-weights models/vikhr-labkovsky-awq /model --force

# Upload ChromaDB
echo "Uploading ChromaDB..."
modal volume put labkovsky-chroma-db chroma_db /chroma_db --force

echo "=== Upload complete ==="
echo ""
echo "Now deploy with: modal deploy modal_deploy.py"

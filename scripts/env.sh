# Sourced by the run scripts: activates the env and pins the Tesla T4 (PCI order, so index 1 on this server).
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hetero
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # less fragmentation on the shared ~10 GB
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs

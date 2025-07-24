# Set the number of GPUs we have
NUM_GPUS=2

# Set environment variable for memory allocation to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# This command automatically handles setting up the distributed environment
python3 -m torch.distributed.launch --nproc_per_node=$NUM_GPUS main.py \
    --listen \
    --multi-user \
    --user-directory=$(pwd)/users/
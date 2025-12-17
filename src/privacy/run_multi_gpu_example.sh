#!/bin/bash

# Example script for distributed training
# This script demonstrates how to use the distributed training functionality

echo "=== Distributed Training Examples ==="
echo ""

# Example 1: DistributedDataParallel training (single node, multiple GPUs)
echo "1. Distributed training on single node with multiple GPUs:"
echo "torchrun --nproc_per_node=4 main.py --distributed --filelist /path/to/filelist.csv --basedir /path/to/images --config config.json"
echo ""

# Example 2: DistributedDataParallel with specific GPU selection
echo "2. Distributed training with specific GPUs:"
echo "CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main.py --distributed --filelist /path/to/filelist.csv --basedir /path/to/images --config config.json"
echo ""

# Example 3: Multi-node distributed training
echo "3. Multi-node distributed training:"
echo "# On node 0:"
echo "torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=\"192.168.1.1\" --master_port=12355 main.py --distributed --filelist /path/to/filelist.csv --basedir /path/to/images --config config.json"
echo "# On node 1:"
echo "torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr=\"192.168.1.1\" --master_port=12355 main.py --distributed --filelist /path/to/filelist.csv --basedir /path/to/images --config config.json"
echo ""

# Example 4: Single GPU training (default behavior)
echo "4. Single GPU training (default):"
echo "python main.py --filelist /path/to/filelist.csv --basedir /path/to/images --config config.json"
echo ""

# Example 5: Debug mode
echo "5. Distributed training with debug information:"
echo "NCCL_DEBUG=INFO torchrun --nproc_per_node=2 main.py --distributed --filelist /path/to/filelist.csv --basedir /path/to/images --config config.json"
echo ""

echo "=== Notes ==="
echo "- DistributedDataParallel provides efficient multi-GPU training"
echo "- Use torchrun for launching distributed training (recommended)"
echo "- Use CUDA_VISIBLE_DEVICES to control which GPUs are used"
echo "- Make sure your batch size is appropriate for the number of GPUs"
echo "- For multi-node training, ensure network connectivity between nodes"
echo "- Only the main process (rank 0) saves checkpoints and logs"
echo "- LOCAL_RANK is automatically set by torchrun - do not pass it manually"

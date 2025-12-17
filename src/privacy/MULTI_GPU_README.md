# Baseline Runs

## Dar Unsupervised 
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main.py --distributed --filelist /vol/ideadata/ed52egek/pycharm/syneverything/datasets/eight_cxr8.csv --basedir /vol/ideadata/ed52egek/pycharm/syneverything/datasets/data --config config_baseline_singlegpuNowmult.json






# Distributed Training Support

This document explains how to use the distributed training functionality that has been added to the Siamese Network training pipeline.

## Overview

The training pipeline supports distributed training using PyTorch's DistributedDataParallel for efficient multi-GPU training across single or multiple nodes.

## Command Line Arguments

### New Arguments

- `--distributed`: Enable DistributedDataParallel training

Note: `local_rank` is automatically handled by `torchrun` via the `LOCAL_RANK` environment variable.

## Usage Examples

### 1. Distributed Training (Single Node)

```bash
# Use multiple GPUs on a single node
torchrun --nproc_per_node=4 main.py --distributed --filelist /path/to/filelist.csv --basedir /path/to/images --config config.json

# Use specific GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main.py --distributed --filelist /path/to/filelist.csv --basedir /path/to/images --config config.json
```

### 2. Multi-Node Distributed Training

```bash
# Multi-node training
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr="192.168.1.1" --master_port=12355 main.py --distributed --filelist /path/to/filelist.csv --basedir /path/to/images --config config.json
```

### 3. Single GPU Training (Default)

```bash
# Default behavior - single GPU
python main.py --filelist /path/to/filelist.csv --basedir /path/to/images --config config.json

# Explicit single GPU
CUDA_VISIBLE_DEVICES=0 python main.py --filelist /path/to/filelist.csv --basedir /path/to/images --config config.json
```

## Key Features

### Distributed Training Setup
- Automatic initialization of distributed process group using NCCL backend
- Proper device assignment for each process
- Local rank management for multi-GPU setups

### Model Wrapping
- **DistributedDataParallel**: Wraps models with `nn.parallel.DistributedDataParallel`
- Handles model unwrapping for checkpoint saving/loading
- Efficient all-reduce communication for gradient synchronization

### Checkpoint Compatibility
- Models are saved without the DistributedDataParallel wrapper
- Loading handles both wrapped and unwrapped model states
- Only the main process (rank 0) saves checkpoints in distributed mode

### Data Loading
- **DistributedDataParallel**: Uses `DistributedSampler` for proper data sharding
- Maintains reproducibility with proper epoch setting
- Automatic data distribution across processes

## Performance Considerations

### DistributedDataParallel Benefits

- **Efficient Communication**: Uses all-reduce communication pattern for gradient synchronization
- **Memory Efficient**: Lower memory usage compared to DataParallel
- **Scalable**: Works well across single nodes and multiple nodes
- **Better Performance**: Optimal for 2+ GPUs with linear scaling

### Batch Size Recommendations

- Keep batch size per GPU constant, total batch size scales with GPU count
- Example: If using batch_size=32 with 4 GPUs, effective batch size becomes 128
- Adjust learning rate accordingly for larger effective batch sizes

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use gradient accumulation
   - Enable mixed precision training

2. **Distributed Training Hangs**
   - Check network connectivity between nodes
   - Ensure all processes can communicate on the specified port
   - Verify master address is accessible
   - Check NCCL backend compatibility

3. **Checkpoint Loading Errors**
   - Ensure checkpoint was saved without DistributedDataParallel wrapper
   - Check device compatibility when loading
   - Verify model architecture matches between save and load

### Debug Mode

```bash
# Enable distributed debug mode
export NCCL_DEBUG=INFO
torchrun --nproc_per_node=2 main.py --distributed --filelist /path/to/filelist.csv --basedir /path/to/images --config config.json
```

## Configuration

The distributed training settings are automatically added to the configuration and can be accessed in your training code:

```python
# In your config or code
config['distributed'] = True  # Enable DistributedDataParallel
# local_rank is automatically set from LOCAL_RANK environment variable
```

## Example Scripts

See `run_multi_gpu_example.sh` for ready-to-use example commands for different scenarios.

## Best Practices

1. **Use torchrun**: Always use `torchrun` for launching distributed training
2. **Monitor Performance**: Use tools like `nvidia-smi` to monitor GPU utilization
3. **Batch Size**: Adjust batch size appropriately for your GPU memory and count
4. **Learning Rate**: Scale learning rate with the number of GPUs for optimal convergence
5. **Checkpointing**: Regularly save checkpoints, especially for long training runs
6. **Error Handling**: Implement proper error handling for distributed training scenarios
7. **Network**: Ensure good network connectivity for multi-node training

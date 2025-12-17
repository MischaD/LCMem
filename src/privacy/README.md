## fpi_mem CLI

Command-line entrypoint for training and evaluating the Patient Verification (Siamese) model.

### Overview
`main.py` loads a JSON config, augments it with CLI flags, and launches `AgentSiameseNetwork`. It expects a CSV listing items and splits, plus a base directory for images. Optionally, a text file of test pairs can override the test set derived from the CSV.

### Requirements
- Python environment with your project dependencies installed
- CUDA (optional) if running on GPU; set `CUDA_VISIBLE_DEVICES` as needed

### Usage
Run from the `privacy/fpi_mem/` directory so relative paths in configs resolve correctly:

```bash
CUDA_VISIBLE_DEVICES=0 \
/vol/ideadata/ed52egek/miniconda/envs/syneverything/bin/python \
  /vol/ideadata/ed52egek/pycharm/syneverything/privacy/fpi_mem/main.py \
  --config_path ./ \
  --config config_dinov3.json \
  --filelist /vol/ideadata/ed52egek/pycharm/syneverything/datasets/alldatasets_balanced.csv \
  --basedir /vol/ideadata/ed52egek/pycharm/syneverything/datasets/data \
  --image_pairs_test /vol/ideadata/ed52egek/pycharm/syneverything/privacy/packhaus/image_pairs/TEST_pairs_cxr8.txt
```

The command above is equivalent to the provided VS Code launch configuration, but executed directly via the CLI.

If you prefer a single line, you can also run:

```bash
CUDA_VISIBLE_DEVICES=0 /vol/ideadata/ed52egek/miniconda/envs/syneverything/bin/python /vol/ideadata/ed52egek/pycharm/syneverything/privacy/fpi_mem/main.py --config_path ./ --config config_dinov3.json --filelist /vol/ideadata/ed52egek/pycharm/syneverything/datasets/alldatasets_balanced.csv --basedir /vol/ideadata/ed52egek/pycharm/syneverything/datasets/data --image_pairs_test /vol/ideadata/ed52egek/pycharm/syneverything/privacy/packhaus/image_pairs/TEST_pairs_cxr8.txt
```

### Arguments
- `--config_path` (default: `./`): Directory where the JSON config file resides. Note this is concatenated with `--config` (not `os.path.join`ed), so include a trailing slash or use `./` as shown.
- `--config` (default: `config.json`): The config filename. Example: `config_dinov3.json`.
- `--filelist` (required): Absolute path to a CSV with at least columns `Split` and `id`.
- `--basedir` (required): Base directory for image paths referenced by the dataset.
- `--image_pairs_test` (optional): A text file of image pairs to use for testing. If provided, it overrides the test set derived from `--filelist`.
- `--experiment_description` (optional): Free-form description; overrides any `experiment_description` in the config if given. Used to name the run archive directory at `./archive/<experiment_description>/`.

### Config Notes
- `main.py` reads `--config_path + --config` and parses it as JSON, then overlays CLI values.
- Ensure the config includes keys expected by `AgentSiameseNetwork` (e.g., architecture, optimization, dataset specifics, and `experiment_description` if you do not pass it on the CLI).

### Working Directory
Prefer running from `privacy/fpi_mem/` so relative assets referenced in configs (if any) resolve as intended:

```bash
cd /vol/ideadata/ed52egek/pycharm/syneverything/privacy/fpi_mem
CUDA_VISIBLE_DEVICES=0 python main.py --config_path ./ --config config_dinov3.json --filelist /vol/ideadata/ed52egek/pycharm/syneverything/datasets/alldatasets_balanced.csv --basedir /vol/ideadata/ed52egek/pycharm/syneverything/datasets/data --image_pairs_test /vol/ideadata/ed52egek/pycharm/syneverything/privacy/packhaus/image_pairs/TEST_pairs_cxr8.txt
```

### Outputs
- A run archive directory is created at `./archive/<experiment_description>/`.
- Training and evaluation are executed via `AgentSiameseNetwork.run()`.



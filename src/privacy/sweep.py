# Import the W&B Python Library and log into W&B
import os
import json
import wandb
from main import run as model_run
from pprint import pprint

def overwrite_cfg(base_cfg, cfg): 
    for k, v in cfg.items():
        if "." in k:
            # handle nested keys like "aug.rotate"
            root, child = k.split(".", 1)
            if root in base_cfg and isinstance(base_cfg[root], dict):
                base_cfg[root][child] = v
            else:
                base_cfg[root] = {child: v}
        else:
            # top-level keys overwrite directly
            base_cfg[k] = v
    return base_cfg

def main():
    with wandb.init() as run:
        # Start from wandb.config (the sweep overrides only search params)
        cfg = dict(run.config)

        # Merge in static config file values
        with open(cfg["config_path"], "r") as f:
            base_cfg = json.load(f)

        base_cfg = overwrite_cfg(base_cfg, cfg)

        pprint(base_cfg)
        # Run your experiment
        base_cfg['experiment_description'] = run.name
        #base_cfg["basedir"] = os.environ.get("TMPDIR", "/tmp")
        print(f"Base dir: {base_cfg['basedir']}")

        os.makedirs(os.path.join('./archive/' , base_cfg['experiment_description']), exist_ok=True)
        score = model_run(base_cfg, wb_run=run)

        # Log your result
        run.log({"score": score})
        print(f"{run.name} with score: {score}")


if __name__ == '__main__':
    main()
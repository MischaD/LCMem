import argparse
import json
import os
import torch
from agent_siamese_network import AgentSiameseNetwork


def main():
    # define an argument parser
    parser = argparse.ArgumentParser('Patient Verification')
    parser.add_argument('--config_path', default='./', help='the path where the config files are stored')
    parser.add_argument('--config', default='config.json', help='the hyper-parameter configuration and experiment settings')
    parser.add_argument("--filelist", type=str, required=True, 
                        help="CSV file with columns 'Split' and 'id'")
    parser.add_argument("--basedir", type=str, required=True,
                        help="Base directory for image paths")
    parser.add_argument("--image_pairs_test", type=str, default=None,
                        help="Optional: Text file with image pairs for testing (overrides filelist for test set)")
    parser.add_argument("--experiment_description", type=str, default=None,
                        help="Optional: experiment description; overrides config if provided")
    parser.add_argument("--distributed", action="store_true", default=False,
                        help="Enable distributed training using DistributedDataParallel")
    parser.add_argument("--unsupervised", action="store_true", default=False,
                        help="Enable unsupervised training by treating all images as different id. Prediction head will be used using Self-Supervision")
    parser.add_argument("--resume_checkpoint", type=str, default=None,
                        help="Path to checkpoint file to resume training from")
    args = parser.parse_args()
    print('Arguments:\n' + '--config_path: ' + args.config_path + '\n--config: ' + args.config)

    # read config
    with open(os.path.join(args.config_path, args.config), 'r') as config:
        config = config.read()

    # parse config
    config = json.loads(config)

    # add command line args to config
    config['filelist'] = args.filelist
    config['basedir'] = args.basedir
    config['image_pairs_test'] = args.image_pairs_test
    config['distributed'] = args.distributed
    config['unsupervised'] = args.unsupervised
    config['resume_checkpoint'] = args.resume_checkpoint
    if args.experiment_description is not None:
        config['experiment_description'] = args.experiment_description

    # create folder to save experiment-related files
    os.makedirs(os.path.join('./archive/' , config['experiment_description']), exist_ok=True)
    SAVINGS_PATH = './archive/' + config['experiment_description'] + '/'
    run(config)


def run(config, wb_run=None):
    # call agent and run experiment
    experiment = AgentSiameseNetwork(config, wb_run=wb_run)
    test_score = experiment.run()
    return test_score


if __name__ == "__main__":
    main()
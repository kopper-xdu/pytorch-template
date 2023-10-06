import sys
import argparse
import train
from omegaconf import OmegaConf


def main():
    cli_config = OmegaConf.from_cli(sys.argv[1:])
    config = OmegaConf.load('config.yaml')
    config = OmegaConf.merge(config, cli_config)

    trainer = train.Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()

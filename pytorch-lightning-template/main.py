import sys, os
from omegaconf import OmegaConf
import lightning.pytorch as pl

from data import ImageNetDataModule
from lightning_module import LitModel


def setup(config):
    pl.seed_everything(42)

    exp_dir = os.path.join('exp', config.exp_name)
    config.default_root_dir = exp_dir
    os.makedirs(exp_dir)
    OmegaConf.save(config, os.path.join(exp_dir, 'config.yaml'))


def main(config):
    trainer = pl.Trainer(**config.trainer_param)

    encoder = model(**config.model_param)
    decoder = None
    lightning_model = LitModel(config, encoder, decoder)

    dm = ImageNetDataModule(**config.data)
    trainer.fit(model=lightning_model, datamodule=dm)

    # trainer.test(model, dataloaders=DataLoader(test_set))


if __name__ == '__main__':
    cli_config = OmegaConf.from_cli(sys.argv[1:])
    config = OmegaConf.load('config.yaml')
    config = OmegaConf.merge(config, cli_config)

    setup(config)
    main(config)
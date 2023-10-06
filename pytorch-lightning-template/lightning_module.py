import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from warmup_scheduler import GradualWarmupScheduler


class LitModel(pl.LightningModule):
    def __init__(self, config, encoder, decoder):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.decoder = decoder

        # self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        x = x.reshape(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss, prog_bar=True)

        # tensorboard = self.logger.experiment
        # tensorboard.add_image()
        # tensorboard.add_histogram(...)
        # tensorboard.add_figure(...)
        
        return loss

    def configure_optimizers(self):
        opt_cfg = self.config.optim
        scheduler_cfg = self.config.scheduler

        optimizer = getattr(torch.optim, opt_cfg.optim_name)(filter(lambda p: p.requires_grad, self.parameters()),
                                                **opt_cfg.optim_param)
        
        scheduler = getattr(torch.optim.lr_scheduler, scheduler_cfg.scheduler_name)(optimizer, **scheduler_cfg.scheduler_param)

        if scheduler_cfg.use_warmup:
            scheduler = GradualWarmupScheduler(optimizer, **scheduler_cfg.warmup_param, after_scheduler=scheduler)
            optimizer.zero_grad()
            optimizer.step()
            scheduler.step()

        return [optimizer], [scheduler]
    
    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        x = x.reshape(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = F.mse_loss(x_hat, x)
        self.log("test_loss", test_loss)

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        x = x.reshape(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        val_loss = F.mse_loss(x_hat, x)
        self.log("val_loss", val_loss)
import lightning.pytorch as pl
from torch.utils.data import random_split, DataLoader
from torchvision import transforms

from .dataset import ImageNet


class ImageNetDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str = "./",
                 train_batch_size: int = 32,
                 val_batch_size: int = 32,
                 test_batch_size: int = 32,
                 num_workers: int = 0,
                 dataset: str | None = None
                 ):
        super().__init__()

        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.dataset = dataset

        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize()
                                             ])
        # self.save_hyperparameters()

    def setup(self, stage: str):
        if stage == "fit":
            mnist_full = ImageNet(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        if stage == "test":
            self.mnist_test = ImageNet(self.data_dir, train=False, transform=self.transform)

        if stage == "predict":
            self.mnist_predict = ImageNet(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.train_batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.val_batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.test_batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.test_batch_size, num_workers=self.num_workers)
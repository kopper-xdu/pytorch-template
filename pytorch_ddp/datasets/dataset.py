from .base_dataset import ImageFolderLMDB


class ImageNet(ImageFolderLMDB):
    def __init__(self, db_path, transform=None, target_transform=None):
        super().__init__(db_path, transform, target_transform)

from ...datasets.builder import DATASETS
from ...datasets.imagenet import ImageNet


@DATASETS.register_module()
class TinyImageNet(ImageNet):
    IMG_EXTENSIONS = ('.jpeg', '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')
    CLASSES = [str(_) for _ in range(200)]

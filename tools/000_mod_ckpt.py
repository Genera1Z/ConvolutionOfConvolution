import os
from collections import OrderedDict

import torch as pt

os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def main():
    """Remove prefix ``backbone`` in the pretrained model's ``state_dict`` and save another copy."""
    ckpt_file = '../shufflenet_v2_batch1024_imagenet_20200812-5bf4721e.pth'
    oupt_file = './shufflenet_v2_batch1024_imagenet-mod.pth'

    tmdl = pt.load(ckpt_file)
    print(tmdl)

    stat_dict = tmdl['state_dict']
    stat_dict2 = OrderedDict()

    for key, value in stat_dict.items():
        key2 = key[9:]
        stat_dict2.update({key2: value})

    tmdl['state_dict'] = stat_dict2

    pt.save(tmdl, oupt_file)
    print(tmdl)
    # tmdl = tmdl.eval()
    # pt.onnx.export(tmdl, (1, 3, 512, 600), open('retina-r18si222mlt1.0-fuse123mode2-all.onnx', 'wb'))


if __name__ == '__main__':
    os.chdir('../')
    main()

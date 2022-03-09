import os

from mmcv.runner.hooks import Hook, HOOKS


@HOOKS.register_module
class DelCkptsHook(Hook):

    def __init__(self, delta=30):
        super(DelCkptsHook, self).__init__()
        self.delta = delta

    def before_train_epoch(self, runner):
        if self.delta >= runner.epoch:
            return
        idxs = runner.epoch - self.delta
        files = [os.path.join(runner.work_dir, f'epoch_{_}.pth') for _ in range(idxs)]
        for file in files:
            if os.path.exists(file) and os.path.isfile(file):
                print(f'remove too old ckpt file `{file}`')
                try:
                    os.remove(file)
                except:
                    print('!!!!!!!!!!!!!! remove failed !!!!!!!!!!!!!!')

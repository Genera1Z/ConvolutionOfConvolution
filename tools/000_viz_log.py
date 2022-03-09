import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_info_from_json(json_file, col_names=('epoch', 'accuracy_top-1', 'accuracy_top-5'), mode='val'):
    assert mode == 'val'
    with open(json_file, 'r') as f:
        lines = f.readlines()
    json_str = '{"infos": [' + ','.join(lines[1:]) + ']}'
    infos_list = json.loads(json_str)['infos']
    infos_pd = pd.DataFrame(infos_list)
    col_mode = infos_pd.get('mode').values
    cols_list = [infos_pd.get(_).values for _ in col_names]
    items_filt = [__ for _, __ in zip(col_mode, list(zip(*cols_list))) if _ == mode]
    cols_dict = {_: __ for _, __ in zip(col_names, list(zip(*items_filt)))}
    return cols_dict


def main():
    col_names = ('epoch', 'accuracy_top-1', 'accuracy_top-5')
    ncol = len(col_names) - 1

    # base_fold = '../!!!!!!!!!!/mmcls.work_dirs/'
    base_fold = './work_dirs/'
    # base_fold = '../!!!!!!!!!!/'
    # dns = ['_hr18-coc-' + _ for _ in [
    #     '22-ttt-ci',
    #     '22-ttt-cico',
    # ]]  + ['_hr18-std', '_hr18-wsc']
    dns = [
        # '_hr18-coc-22-ttf-ci',  # 61.05
        # '_hr18-coc-22-ttf-cico',  # 61.25
        # '_hr18-coc-22-ttt-ci',  # 61.79
        # '_hr18-coc-22-ttt-ci-both',  # 62.68  TODO * selected
        # '_hr18-coc-22-ttt-cico',  # 60.94
        # '_hr18-coc-22-ttt-co',  # very bad (0422)
        # '_hr18-coc-22-ttt-co-both',  # 61.92 (0422)  # TODO ?
        # '_hr18-coc-22-ttt-none',  # 59.47 (0422)
        # '_hr18-coc-22-ttt-none-both',  # 60.89 (0422)
        # '_hr18-std',  # 59.97
        # '_hr18-wsc'  # 61.76

        ### TODO hr32

        # '_r18-coc-22-ttt-ci',  # 56.55
        # '_r18-coc-22-ttt-ci-new_dilat',  # 56.35
        # '_r18-coc-22-ttt-ci-both',  # 56.43
        # '_r18-coc-22-ttt-ci-both-new_dilat',  # 56.42
        # '_r18-coc-22-ttt-cico',  # 54.86
        # '_r18-coc-22-ttt-cico-both',  # 55.37
        # '_r18-coc-22-ttt-co',  # 50.91 (0422)
        # '_r18-coc-22-ttt-co-both',  # 56.24 (0422)
        # '_r18-coc-22-ttt-none',  # 57.26 (0422)
        # '_r18-coc-22-ttt-none-both',  # 57.45 (0422)
        # '_r18-coc-22-ttt-none-both-larger_dilat',  # 57.67 (0422)  # TODO selected
        # '_r18-coc-24-ttt-none-both',  # 56.97 (0422)
        # '_r18-std',  # 56.17
        # '_r18-wsc'  # 57.19

        # '_r50-coc-22-ttf-ci',  # 60.21
        # '_r50-coc-22-ttf-cico',  # 58.61 | 60.00
        # '_r50-coc-22-ttt-ci',  # 60.21  # TODO *
        # '_r50-coc-22-ttt-ci-new_dilat',  # 60.09
        # '_r50-coc-22-ttt-ci-both',  # 59.84
        # '_r50-coc-22-ttt-ci-both-new_dilat',  # 60.04
        # '_r50-coc-22-ttt-cico',  # 59.38 | 60.30
        # '_r50-coc-22-ttt-co',  # 57.04 (0422)
        # '_r50-coc-22-ttt-co-both',  # 60.14 (0422)
        # '_r50-coc-22-ttt-none',  # 59.70 (0422)
        # '_r50-coc-22-ttt-none-both',  # 59.60 (0422)
        # '_r50-coc-22-ttt-none-both-larger_dilat',  # 60.26 (0422)  # TODO selected
        # '_r50-coc-24-ttt-none-both',  # 59.63 (0422)
        # '_r50-std',  # 59.02
        # '_r50-wsc'  # * 60.31

        # '_mb2-coc-22-ttf-ci',
        # '_mb2-coc-22-ttf-ci-f',  # 49.17  == ttf-ci-f-both  XXX * (expected)
        # '_mb2-coc-22-ttf-ci-ff',
        # '_mb2-coc-22-ttt-ci',
        # '_mb2-coc-22-ttt-ci-f',  # 46.68
        # '_mb2-coc-22-ttt-ci-f-both',  # ? similar to ttf-ci-f
        # '_mb2-std',  # * 50.69
        # '_mb2-wsc'  # 45.43

        ### for DWC, share-none=share-cico=share-co; only share-ci is best

        # '_mb3-coc-22-ttf-ci-f',  # 34.04 (not finish)
        # '_mb3-coc-22-ttf-ci-f-new_dilat',  # 40.77 | 41.36 (0422)  # TODO ?
        # '_mb3-coc-22-ttf-ci-f-new_dilat-larger_dilat',  # 40.99 (0422)  # TODO selected
        # '_mb3-coc-24-ttf-ci-f-new_dilat'  # 39.98 (0422)
        # '_mb3-coc-22-ttt-ci-f',  # 39.10
        # '_mb3-coc-22-ttt-ci-f-both',  # 41.23
        # '_mb3-coc-22-ttt-ci-f-both-new_dilat',  # 40.81 (0422)
        # '_mb3-std',  # 41.85
        # '_mb3-wsc'  # 32.12 (not finish)

        # '_sf2-coc-22-ttf-ci',
        # '_sf2-coc-22-ttf-ci-f',  # * 52.18  == ttf-ci-f-both  # TODO *
        # '_sf2-coc-22-ttf-ci-f-new_dilat',  # * 52.18 | 51.85 (0422)  == ttf-ci-f-both  # TODO selected
        # '_sf2-coc-24-ttf-ci-f-new_dilat',  # 51.95 (0422)  TODO *
        # '_sf2-coc-22-ttt-ci',
        # '_sf2-coc-22-ttt-ci-f',
        # '_sf2-coc-22-ttt-ci-f-both',  # 51.83
        # '_sf2-coc-22-ttt-ci-f-both-new_dilat',  # 51.94 (0422)
        # '_sf2-std',  # 51.74
        # '_sf2-wsc'  # 48.16
    ]

    fig, axes = plt.subplots(ncol)
    _makeshift = '-tft-', '--'  # XXX
    _start1 = 31  # XXX 31
    _start2 = -30  # XXX 61

    for dn in dns:
        # find json log file, read log by json
        sub_fold = os.path.join(base_fold, dn)
        json_files = glob.glob(sub_fold + '/*.log.json')
        assert len(json_files) == 1
        json_file = json_files[0]
        cols_dict = read_info_from_json(json_file, col_names)
        # viz by items
        for i in range(ncol):
            if _makeshift[0] in dn:  # XXX
                linestyle = _makeshift[1]
            else:
                linestyle = '-'
            xs, ys = cols_dict[col_names[0]], cols_dict[col_names[i + 1]]
            xs, ys = xs[_start1:], ys[_start1:]
            _mean = np.mean(ys[_start2:])
            _std = np.std(ys[_start2:])
            axes[i].plot(xs, ys, label=f'{dn[5:]} {_mean:.2f} {_std:.2f}', linestyle=linestyle)

    [_.legend() for _ in axes]
    # fig.legend(loc='upper left')
    # axes[0].legend(loc='lower left')
    plt.show()
    # fig.show(); fig.waitforbuttonpress()


if __name__ == '__main__':
    os.chdir('../')
    main()

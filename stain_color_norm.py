# -*- coding: UTF-8 -*-
import subprocess
import pathlib
import click
import multiprocessing
from glob import glob
import os

def transfer_tiatoolbox(source_input, target_input, output_dir, file_types):
    sub_p = subprocess.Popen(
        '/home/bio19/anaconda3/bin/tiatoolbox stainnorm --source_input {} --target_input {} '
        ' --method "macenko" --output_dir {} --file_types {}'.format(
            source_input, target_input, output_dir, file_types),
        shell=True,
        stderr=subprocess.PIPE
    )
    sub_p.wait()

    return True


@click.command()
@click.option(
    "--source_input",
    help="input path to the source image or a directory of source images", default='tmp/data/tiles')
@click.option("--target_input", help="input path to the target image", default='asset/Template.png')
@click.option("--output_dir", help="output directory for normalisation", default='tmp/data/tiles_color_normalized')
@click.option(
    "--file_types",
    help="file types to capture from directory",
    default="*orig.png"
    )
def batch_cn(source_input, target_input, output_dir, file_types='*orig.png'):
    """External package tiatoolbox,
       multi-threaded thread pool automatically completes stainnorm.
    """
    if pathlib.Path(source_input).is_dir():
        paths = pathlib.Path(source_input).glob('*')
        pool_multi = multiprocessing.Pool(processes=4)
        for path in paths:
            name = '-'.join(path.stem.split('-')[:3])
            pathlib.Path(output_dir).joinpath(name).mkdir(parents=True, exist_ok=True)
            list1 = glob(os.path.join(os.path.join('tmp/data/tiles','{}?*'.format(name)),'*orig.png'))
            list2 = glob(os.path.join(os.path.join('tmp/data/tiles_color_normalized',name),'*.png'))
            if len(list1) == len(list2):
                print(name,'return')
                continue
            pool_multi.apply(transfer_tiatoolbox,
                             (path,
                              target_input,
                              pathlib.Path(output_dir).joinpath(name),
                              file_types))

        pool_multi.close()
        pool_multi.join()
    elif pathlib.Path(source_input).is_file():
        name = '-'.join(pathlib.Path(source_input).parent.stem.split('-')[:3])
        pathlib.Path(output_dir).joinpath(name).mkdir(parents=True, exist_ok=True)
        transfer_tiatoolbox(pathlib.Path(source_input), target_input,
                            pathlib.Path(output_dir).joinpath(name), file_types)
    else:
        raise FileNotFoundError


if __name__ == '__main__':
    batch_cn()

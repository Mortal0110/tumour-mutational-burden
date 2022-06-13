import argparse
import openslide
import os
import sys
from glob import glob
from os.path import join
from preprocessing.load_xml import load_xml
from preprocessing.get_ret_ifo import get_ret_ifo
from utils.common import logger


path_wd = os.path.dirname(sys.argv[0])
sys.path.append(path_wd)
if not path_wd == '':
    os.chdir(path_wd)
need_save = False


def prepare_data(images_dir_root, images_dir_split, size_square):
    num_name = 0
    # img_data = []
    # pos = 0

    image_dir_list = glob(join(images_dir_root, r'*/'))
    for image_dir in image_dir_list:
        xml_files = glob(join(image_dir, '*.xml'))
        for index_xml in range(len(xml_files)):
            num_name += 1
            logger.info("xml_files:".format(xml_files[index_xml]))

            xy_list = load_xml(xml_files[index_xml])
            if os.path.exists(xml_files[index_xml].split('xml')[0]+'svs'):
                image_address = xml_files[index_xml].split('xml')[0] + 'svs'
            else:
                continue
            slide = openslide.open_slide(image_address)
            # image_large = \
            get_ret_ifo(xy_list, slide, image_address, images_dir_split,
                                      size_square, size_square, 3, 0.3)

            # for i in range(len(image_large)):
            #     image_small = image_large[i]
            #     for j in range(len(image_small)):
            #         img_data.append(image_small[j])
            #         pos += 1
            # print(num_name)
    logger.info('tiles are done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='svs to tiles')
    parser.add_argument('--slide_image_root', type=str, default="tmp/data/images")
    parser.add_argument('--tiles_image_root', type=str, default="tmp/data/tiles")
    parser.add_argument('--size_square', type=int, default=512)
    parser.add_argument('--prepare_types', type=str, default=".svs")
    args = parser.parse_args()

    logger.info('Processing svs images to tiles')
    assert args.prepare_types == ".svs", "svs slide support only"
    prepare_data(args.slide_image_root, args.tiles_image_root, args.size_square)

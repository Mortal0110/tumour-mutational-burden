# -*- coding: UTF-8 -*-

import os
import random
import shutil
from glob import glob
import pandas as pd
import argparse
import yaml
from utils.common import logger
from sklearn.model_selection import KFold
import os

with open(os.path.join(os.getcwd(), 'config/config.yml'), 'r', encoding='utf8') as fs:
    cfg = yaml.load(fs, Loader=yaml.FullLoader)
RATIO_TRAIN = cfg['ratio_train']
RATIO_TEST = cfg['ratio_test']


def remove_svs(svs_list):
    return_list = set()
    for file in svs_list:
        file_name = '-'.join(file.split('-')[:3])
        if file_name not in return_list:
            return_list.add(file_name)
    return list(return_list)


def remove_unmatched_elements(seq1, seq2, img):
    for i in seq1[::-1]:
        imgs_len = len(glob(os.path.join(img, i, '*')))
        if (i not in seq2) or (imgs_len < 20):
            seq1.remove(i)
    return seq1


def split_train_and_test_set(seq):
    train_set = seq[:int(len(seq) * RATIO_TRAIN)]
    test_set = seq[int(len(seq) * RATIO_TRAIN):]
    return train_set, test_set


def main(img, label, shuffle=True):

    img_dir = glob(os.path.join(img, '*/'))
    xml_file_seq = [img.split('/')[-2] for img in img_dir]
    df = pd.read_csv(label, usecols=["file_name", "labels"])
    high_label_seq = [getattr(row, 'file_name') for row in df.itertuples() if getattr(row, 'labels') == 1]
    low_label_seq = [getattr(row, 'file_name') for row in df.itertuples() if getattr(row, 'labels') == 0]
    # classify by person
    high_label_seq = remove_svs(high_label_seq)
    low_label_seq = remove_svs(low_label_seq)

    high_label_matched_seq = remove_unmatched_elements(high_label_seq, xml_file_seq, img)
    low_label_matched_seq = remove_unmatched_elements(low_label_seq, xml_file_seq, img)

    X = high_label_matched_seq+low_label_matched_seq
    y = [1 for _ in range(len(high_label_matched_seq))] + [0 for _ in range(len(low_label_matched_seq))]
    # 5 KFold
    kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
    folder = 0
    for train_index, test_index in kf.split(X,y):
        print('KFold:{}'.format(folder))
        high_train_set = [X[j] for j in train_index if y[j] == 1]
        high_test_set = [X[j] for j in test_index if y[j] == 1]
        low_train_set = [X[j] for j in train_index if y[j] == 0]
        low_test_set = [X[j] for j in test_index if y[j] == 0]
        high_cn_files = []
        low_cn_files = []		
        for high in high_train_set:
            high_cn_files.extend(glob(os.path.join(img, high, '*orig.png')))
        for low in low_train_set:
            low_cn_files.extend(glob(os.path.join(img, low, '*orig.png')))
        # subsampling
        logger.info('# of high tmb-index tile is: {0}, # of low tmb-index tile is: {1}'.format(len(high_cn_files),
                                                                                           len(low_cn_files)))
        logger.info('subsampling...')
        if len(low_cn_files) > len(high_cn_files):
            index = random.sample(range(0, len(low_cn_files)), len(high_cn_files))
            if len(set(index)) != len(index):
                raise ValueError('Repeated data is obtained during subsampling')
            low_cn_files = [low_cn_files[i] for i in index]
        elif len(low_cn_files) < len(high_cn_files):
            index = random.sample(range(0, len(high_cn_files)), len(low_cn_files))
            if len(set(index)) != len(index):
                raise ValueError('Repeated data is obtained during subsampling')
            high_cn_files = [high_cn_files[i] for i in index]
        else:
            pass

        # Create data directory structure
        for data in ['train'+str(folder), 'test'+str(folder)]:
            for label in ['0', '1']:
                if not os.path.exists(os.path.join(img, '..', data, label)):
                    os.makedirs(os.path.join(img, '..', data, label))


        # Data partition
        for file in low_cn_files:
            shutil.copy(file, os.path.join(img, '..', 'train'+str(folder), '0', file.split('/')[-1]))
        for file in high_cn_files:
            shutil.copy(file, os.path.join(img, '..', 'train'+str(folder), '1', file.split('/')[-1]))
        for file in low_test_set:
            for img_low in glob(os.path.join(img, file, '*orig.png')):
                shutil.copy(img_low, os.path.join(img, '..', 'test'+str(folder), '0', img_low.split('/')[-1]))
        for file in high_test_set:
            for img_high in glob(os.path.join(img, file, '*orig.png')):
                shutil.copy(img_high, os.path.join(img, '..', 'test'+str(folder), '1', img_high.split('/')[-1]))

        folder += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script',
                                     epilog="authorized by geneis ")
    parser.add_argument('--stained_tiles_home', type=str, default="tmp/data/tiles_color_normalized/")
    parser.add_argument('--label_dir_path', type=str, default="tmp/data/labels/reg_tmb.csv")
    args = parser.parse_args()
    main(args.stained_tiles_home, args.label_dir_path)


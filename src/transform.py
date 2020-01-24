#coding=utf-8

"""
Provides options via the command line to perform project tasks
* `--path`: path to the raw dataset
* `--type`: type of the data (train, validation, or test)
* `--transform`: transform dataset 
* `--sample`: visualize sample from transformed dataset
"""

import argparse
import cv2
import h5py
import os
import pathlib
import sys
import json
from multiprocessing import Pool
from functools import partial

import preprocess as prep

INPUT_SIZE = (2048, 128, 1)
SAMPLE_SIZE = 50
PREDICTION_LINE_CODE = 'P'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=False)
    parser.add_argument("--type", type=str, required=True)
    parser.add_argument("--sample", action="store_true", default=False)
    parser.add_argument("--transform", action="store_true", default=False)
    args = parser.parse_args()

    raw_path = os.path.join(args.path) if args.path else ''
    output_path = os.path.join('..', 'data', args.type)
    prediction_file = os.path.join('..', 'evaluation', 'predictions_{}.txt'.format(args.type))

    if args.transform:
        assert os.path.exists(raw_path)
        print('Start transforming data')
        os.makedirs(os.path.join(output_path), exist_ok=True)

        print('INFO: Get image data and labels')
        image_paths = [{'path':str(item), 'name':item.name} for item in pathlib.Path(raw_path).glob('*') if item.is_file() and not str(item.name).endswith('json')]
        assert len(image_paths) > 0
        label_path = [str(item) for item in pathlib.Path(raw_path).glob('*') if item.is_file() and str(item.name).endswith('json')]
        assert len(label_path) == 1
        labels = json.load(open(label_path[0]))

        dataset = {"image_path":[], "label":[]}
        for item in image_paths:
            dataset['image_path'].append(item['path'])
            dataset['label'].append(labels[item['name']].encode())

        print('INFO: Preprocessing images')
        pool = Pool()
        dataset['image'] = pool.map(partial(prep.preprosess_raw, input_size=INPUT_SIZE, save_type=args.type), dataset['image_path'])
        pool.close()
        pool.join()

        with open(os.path.join('..', 'data', '{}.json'.format(args.type)), 'w') as f:
            json.dump(labels, f)

        # print('INFO: Store data in {}'.format(output_path))
        # with h5py.File(output_path, "w") as hf:
        #     hf.create_dataset("image", data=dataset['image'], compression="gzip", compression_opts=9)
        #     hf.create_dataset("label", data=dataset['label'], compression="gzip", compression_opts=9)

    elif args.sample:
        # with h5py.File(output_path, 'r') as hf:
        #     images = hf['image'][:SAMPLE_SIZE]
        #     labels = hf['label'][:SAMPLE_SIZE]

        labels = json.load(open(os.path.join('..', 'data', '{}.json'.format(args.type))))
        image_paths = [str(item) for item in pathlib.Path(os.path.join(output_path)).glob('*') if item.is_file()][:SAMPLE_SIZE]

        if os.path.isfile(prediction_file):
            with open(prediction_file, 'r') as f:
                predictions = [line[len(PREDICTION_LINE_CODE):] for line in f if line.startswith(PREDICTION_LINE_CODE)]
        else:
            predictions = [''] * SAMPLE_SIZE

        for i in range(SAMPLE_SIZE):
            image = cv2.imread(image_paths[i], cv2.IMREAD_GRAYSCALE)
            file_name = pathlib.Path(image_paths[i]).name
            print("Image shape:\t{}".format(image.shape))
            print("Label:\t{}".format(labels[file_name]))
            print("Predict:\t{}\n".format(predictions[i]))

            cv2.imshow("img", prep.adjust_to_see(image))
            cv2.waitKey(0)

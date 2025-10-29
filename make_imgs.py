# -*- coding: utf-8 -*-

import os
import pickle
import logging

# set logging level - for logging to file add: filename='myapp.log',
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='\t\t%(asctime)s - %(levelname)s - %(message)s')

path_to_pkl = "./giga_pickles"
image_summaries_pkl = os.path.join(path_to_pkl,"image_summaries_pkl.pkl")
print(f"\t\t\t{image_summaries_pkl}")

imgs_pkl = os.path.join(path_to_pkl,"imgs_pkl.pkl")

def load_image_summaries():
    with open(image_summaries_pkl, 'rb') as inp:
        print(f"\t\tfile loaded - ok: {image_summaries_pkl}")
        return pickle.load(inp)

image_summaries = load_image_summaries()

# Если есть изображения, выполняем их описание
imgs = []
if image_summaries:
    for image in image_summaries:
        imgs.append(image.get('image_content'))

    with open(imgs_pkl, 'wb') as outp:
        pickle.dump(imgs, outp, pickle.HIGHEST_PROTOCOL)

logger.info(f'length image_summaries = {len(image_summaries)} length imgs: = {len(imgs)}')
if len(imgs) > 0:
    print(imgs[:3], flush=True)
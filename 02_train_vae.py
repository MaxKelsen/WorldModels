#python 02_train_vae.py --new_model

from vae.arch128_rect import VAE
import argparse
import numpy as np
import config

import time
import scipy.misc
import pandas as pd
import cv2

def downscale_images(img_array):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    print("Input image array has shape: " + str(img_array.shape))
    i = 0
    resized = []
    for img in img_array:
        res = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
        resized.append(res)
        #scipy.misc.imsave('images/img' + str(i) + '.jpg', img)
        #scipy.misc.imsave('images/res' + str(i) + '.jpg', res)
        i += 1
    res_array = np.stack(resized, axis=0)
    print("Resized to new array of shape: " + str(res_array.shape))
    return img_array

def main(args):
    start_batch = args.start_batch
    max_batch = args.max_batch
    new_model = args.new_model

    vae = VAE()

    if not new_model:
        try:
            vae.set_weights('./vae/weights.h5')
        except:
            print("Either set --new_model or ensure ./vae/weights.h5 exists")
            raise

    train = pd.read_csv('sonic-train.csv')
    validation = pd.read_csv('sonic-validation.csv')

    for row in train.iterrows():
        game = row[1]['game']
        state = row[1]['state']

        for batch_num in range(0, 10):
            try:
                data = np.load('../retro-movies/data/obs_data_' + game + '_' + state + '_' + str(batch_num) + '.npy')
            except:
                #print('no data found for batch number {}'.format(batch_num))
                break
            
            data = np.array([item for obs in data for item in obs]) 
            if(data.shape[0] == 0):
                break
            print(game + "_" + state +  ", " + str(batch_num))
            print(data.shape)
	    #data = downscale_images(data)  # scale images to 64*64
            vae.train(data)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=('Train VAE'))
  parser.add_argument('--start_batch', type=int, default = 0, help='The start batch number')
  parser.add_argument('--max_batch', type=int, default = 0, help='The max batch number')
  parser.add_argument('--new_model', action='store_true', help='start a new model from scratch?')
  args = parser.parse_args()

  main(args)

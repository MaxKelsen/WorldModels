#python 02_train_vae.py --new_model

from vae.arch128_rect import VAE
import argparse
import numpy as np
import config

import time
import scipy.misc

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
        scipy.misc.imsave('images/img' + str(i) + '.jpg', img)
        scipy.misc.imsave('images/res' + str(i) + '.jpg', res)
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

  for batch_num in range(start_batch, max_batch + 1):
    print('Building batch {}...'.format(batch_num))
    first_item = True

    for env_name in config.train_envs:
      try:
        new_data = np.load('./data/obs_data_' + env_name + '_' + str(batch_num) + '.npy')
        if first_item:
          data = new_data
          first_item = False
        else:
          data = np.concatenate([data, new_data])
        print('Found {}...current data size = {} episodes'.format(env_name, len(data)))
      except:
        pass

    if first_item == False: # i.e. data has been found for this batch number
      # here data is every image in batch with shape (6000, 224, 320, 3)
      data = np.array([item for obs in data for item in obs]) 
      data = downscale_images(data)  # scale images to 64*64
      vae.train(data)
    else:
      print('no data found for batch number {}'.format(batch_num))

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=('Train VAE'))
  parser.add_argument('--start_batch', type=int, default = 0, help='The start batch number')
  parser.add_argument('--max_batch', type=int, default = 0, help='The max batch number')
  parser.add_argument('--new_model', action='store_true', help='start a new model from scratch?')
  args = parser.parse_args()

  main(args)

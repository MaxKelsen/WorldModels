#python 03_generate_rnn_data.py

from vae.arch_fullsize import VAE
import argparse
import config
import numpy as np

import pandas as pd

def main(args):

    start_batch = args.start_batch
    max_batch = args.max_batch

    vae = VAE()

    try:
        vae.set_weights('./vae/weights.h5')
    except:
        print("./vae/weights.h5 does not exist - ensure you have run 02_train_vae.py first")
        raise

    train = pd.read_csv('sonic-train.csv')
    validation = pd.read_csv('sonic-validation.csv')

    for row in train.iterrows():
        game = row[1]['game']
        state = row[1]['state']

        for batch_num in range(0, 10):
            try:
                obs_data = np.load('../retro-movies/data/obs_data_' + game + '_' + state + '_' + str(batch_num) + '.npy')
                action_data = np.load('../retro-movies/data/action_data_' + game + '_' + state + '_' + str(batch_num) + '.npy')
            except:
                #print('no data found for batch number {}'.format(batch_num))
                break
            
            if(obs_data.shape[0] == 0):
                break
            print(game + "_" + state +  ", " + str(batch_num))
            print(obs_data.shape)

            rnn_input, rnn_output = vae.generate_rnn_data(obs_data, action_data)
            np.save('./data/rnn_input_' + game + '_' + state + '_' + str(batch_num), rnn_input)
            np.save('./data/rnn_output_' + game + '_' + state + '_' + str(batch_num), rnn_output)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=('Generate RNN data'))
  parser.add_argument('--start_batch', type=int, default = 0, help='The start batch number')
  parser.add_argument('--max_batch', type=int, default = 0, help='The max batch number')

  args = parser.parse_args()

  main(args)

#python 04_train_rnn.py --new_model

from rnn.arch_fullsize import RNN
import argparse
import numpy as np

import pandas as pd

def main(args):
    
    start_batch = args.start_batch
    max_batch = args.max_batch
    new_model = args.new_model

    rnn = RNN()

    if not new_model:
        try:
          rnn.set_weights('./rnn/weights.h5')
        except:
          print("Either set --new_model or ensure ./rnn/weights.h5 exists")
          raise

    train = pd.read_csv('sonic-train.csv')
    validation = pd.read_csv('sonic-validation.csv')

    rownum = 0
    for row in train.iterrows():
        game = row[1]['game']
        state = row[1]['state']

        for batch_num in range(0, 10):
            print(game + "_" + state +  ", " + str(batch_num))
            try:
                new_rnn_input = np.load('./data/rnn_input_' + game + '_' + state + '_' + str(batch_num) + '.npy') 
                new_rnn_output = np.load('./data/rnn_output_' + game + '_' + state + '_' + str(batch_num) + '.npy')
            except:
                #print('no data found for batch number {}'.format(batch_num))
                break

            if (batch_num == 0):
                rnn_input = new_rnn_input
                rnn_output = new_rnn_output
            else:
                rnn_input = np.concatenate([rnn_input, new_rnn_input])
                rnn_output = np.concatenate([rnn_output, new_rnn_output])
        
        if (rownum == 0):
            rnn_input_all = rnn_input
            rnn_output_all = rnn_output
        else:
            rnn_input_all = np.concatenate([rnn_input_all, rnn_input])
            rnn_output_all = np.concatenate([rnn_output_all, rnn_output])
    
        rownum += 1
    
    print (rnn_input_all.shape)
    print (rnn_output_all.shape)
    rnn.train(rnn_input_all, rnn_output_all)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Train RNN'))
    parser.add_argument('--start_batch', type=int, default = 0, help='The start batch number')
    parser.add_argument('--max_batch', type=int, default = 0, help='The max batch number')
    parser.add_argument('--new_model', action='store_true', help='start a new model from scratch?')

    args = parser.parse_args()

    main(args)

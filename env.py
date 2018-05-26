import numpy as np
#import gym
# from custom_envs.car_racing import CarRacing

import os
os.environ['DYLD_FALLBACK_LIBRARY_PATH'] = '/lib:/usr/lib:/usr/bin/lib:/' + os.environ['DYLD_FALLBACK_LIBRARY_PATH'];

import retro
import pyglet

def make_env(env_name, seed=-1, render_mode=False):
  if env_name == 'car_racing':
    env = CarRacing()
    if (seed >= 0):
      env.seed(seed)
  elif env_name == 'sonic':
    env = retro.make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act3')
  else:
    print("couldn't find this env")

  return env

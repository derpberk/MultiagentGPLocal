#!/bin/sh

# EXPERIMENTOS CON WQP 

# Experimentos ALGAE_BLOOM con REWARD MU
python Training/TrainLocalA2C.py --N 3 --R changes_mu --GT shekel --GPU $1
python Training/TrainLocalA2C.py --N 2 --R changes_mu --GT shekel --GPU $1
python Training/TrainLocalA2C.py --N 1 --R changes_mu --GT shekel --GPU $1 

# Experimentos ALGAE_BLOOM con REWARD SIGMA
python Training/TrainLocalA2C.py --N 3 --R changes_sigma --GT shekel --GPU $1
python Training/TrainLocalA2C.py --N 2 --R changes_sigma --GT shekel --GPU $1
python Training/TrainLocalA2C.py --N 1 --R changes_sigma --GT shekel --GPU $1 


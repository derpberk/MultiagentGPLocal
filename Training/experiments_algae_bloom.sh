
# Experimentos ALGAE_BLOOM con REWARD MU
python Training/TrainLocalDQL.py --N 3 --R changes_mu --GT algae_bloom --GPU $1
python Training/TrainLocalDQL.py --N 2 --R changes_mu --GT algae_bloom --GPU $1
python Training/TrainLocalDQL.py --N 1 --R changes_mu --GT algae_bloom --GPU $1 

# Experimentos ALGAE_BLOOM con REWARD SIGMA
python Training/TrainLocalDQL.py --N 3 --R changes_sigma --GT algae_bloom --GPU $1
python Training/TrainLocalDQL.py --N 2 --R changes_sigma --GT algae_bloom --GPU $1
python Training/TrainLocalDQL.py --N 1 --R changes_sigma --GT algae_bloom --GPU $1 


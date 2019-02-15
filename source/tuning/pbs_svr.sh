#PBS -N SVR
#PBS -l select=1:ncpus=56:nodetype=n56
#PBS -l walltime=336:00:00

module load python/3.5.4

cd predicting_high_low_TG/
source env3.5/bin/activate

cd source/

# uncomment to run

# make run_tuning_svr input_file=../../data/clean/oxides_Tg_train.csv output_folder=../../result/ max_iter=500 seed=500 n_jobs=50 data_tag=tg
# make run_tuning_svr input_file=../../data/clean/oxides_ND300_train.csv output_folder=../../result/ max_iter=500 seed=500 n_jobs=50 data_tag=nd300
# make run_tuning_svr input_file=../../data/clean/oxides_Tliquidus_train.csv output_folder=../../result/ max_iter=500 seed=500 n_jobs=50 data_tag=tl

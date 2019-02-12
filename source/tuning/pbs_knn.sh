#PBS -N k-NN
#PBS -l select=1:ncpus=56:nodetype=n56
#PBS -l walltime=336:00:00

module load python/3.5.4

cd predicting_high_low_TG/
source env3.5/bin/activate

cd source/
make run_tuning_knn

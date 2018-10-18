#$1 arg 1

#PBS -N glass
#PBS -l select=1:ncpus=56:nodetype=n56
#PBS -l walltime=336:00:00
#PBS -m abe
#PBS -M e.alcobaca@gmail.com

module load python/3.5.4
source predicting_high_low_TG_2/predicting_high_low_TG/env3.5/bin/activate
cd predicting_high_low_TG_2/predicting_high_low_TG/source/range

python run.py --n_cpus=27 -e 3

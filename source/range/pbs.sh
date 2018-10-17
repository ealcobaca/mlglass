#$1 arg 1

#PBS -N glass
#PBS -l select=5:ncpus=40:nodetype=n40
#PBS -l walltime=336:00:00
#PBS -m abe
#PBS -M e.alcobaca@gmail.com

source predicting_high_low_TG/env3.5/bin/activate
cd predicting_high_low_TG/source/range

python run.py --n_cpus=81 -e 3

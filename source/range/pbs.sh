#$1 arg 1

#PBS -N glass
#PBS -l select=1:ncpus=56:nodetype=n56
#PBS -l walltime=336:00:00
#PBS -m abe
#PBS -M e.alcobaca@gmail.com

source predicting_high_low_TG/exp3.5/bin/activate
cd /home/alcobaca/predicting_high_low_TG/source/range

printf "$(pwd)\n"

python run.py --n_cpus=56 -e 3

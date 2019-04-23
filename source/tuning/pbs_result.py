#PBS -N DT
#PBS -l select=1:ncpus=30:mem=30GB
#PBS -l walltime=24:00:00

module load python/3.5.4

cd /lustre/alcobaca/predicting_high_low_TG/
source env3.5/bin/activate

cd source/


# uncomment to run
make generate_logs


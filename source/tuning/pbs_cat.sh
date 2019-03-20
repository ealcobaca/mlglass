#PBS -N CAT
#PBS -l select=1:ncpus=40:nodetype=n40
#PBS -l walltime=336:00:00

module load python/3.5.4

cd /lustre/andre/predicting_high_low_TG/
source env3.5/bin/activate

cd source/

# uncomment to run
make run_tuning_catboost n_jobs=38 outer_fold=1  tuning_seed=1
make run_tuning_catboost n_jobs=38 outer_fold=2  tuning_seed=2
make run_tuning_catboost n_jobs=38 outer_fold=3  tuning_seed=3
make run_tuning_catboost n_jobs=38 outer_fold=4  tuning_seed=4
make run_tuning_catboost n_jobs=38 outer_fold=5  tuning_seed=5

make run_tuning_catboost n_jobs=38 outer_fold=6  tuning_seed=6
make run_tuning_catboost n_jobs=38 outer_fold=7  tuning_seed=7
make run_tuning_catboost n_jobs=38 outer_fold=8  tuning_seed=8
make run_tuning_catboost n_jobs=38 outer_fold=9  tuning_seed=9
make run_tuning_catboost n_jobs=38 outer_fold=10 tuning_seed=10


# 5 expriment
# make run_tuning_catboost input_file=../../data/clean/oxides_ND300_train.csv output_folder=../../result/ max_iter=100 seed=100 n_jobs=38 data_tag=nd300
# make run_tuning_catboost input_file=../../data/clean/oxides_ND300_train.csv output_folder=../../result/ max_iter=100 seed=200 n_jobs=38 data_tag=nd300
# make run_tuning_catboost input_file=../../data/clean/oxides_ND300_train.csv output_folder=../../result/ max_iter=100 seed=300 n_jobs=38 data_tag=nd300
# make run_tuning_catboost input_file=../../data/clean/oxides_ND300_train.csv output_folder=../../result/ max_iter=100 seed=400 n_jobs=38 data_tag=nd300
# make run_tuning_catboost input_file=../../data/clean/oxides_ND300_train.csv output_folder=../../result/ max_iter=100 seed=500 n_jobs=38 data_tag=nd300
# ##########################################################################
#
# # 5 expriment
# make run_tuning_catboost input_file=../../data/clean/oxides_Tliquidus_train.csv output_folder=../../result/ max_iter=100 seed=100 n_jobs=38 data_tag=tl
# make run_tuning_catboost input_file=../../data/clean/oxides_Tliquidus_train.csv output_folder=../../result/ max_iter=100 seed=200 n_jobs=38 data_tag=tl
# make run_tuning_catboost input_file=../../data/clean/oxides_Tliquidus_train.csv output_folder=../../result/ max_iter=100 seed=300 n_jobs=38 data_tag=tl
# make run_tuning_catboost input_file=../../data/clean/oxides_Tliquidus_train.csv output_folder=../../result/ max_iter=100 seed=400 n_jobs=38 data_tag=tl
# make run_tuning_catboost input_file=../../data/clean/oxides_Tliquidus_train.csv output_folder=../../result/ max_iter=100 seed=500 n_jobs=38 data_tag=tl
# ##########################################################################
#
# # 5 expriment
# make run_tuning_catboost input_file=../../data/clean/oxides_Tg_train.csv output_folder=../../result/ max_iter=100 seed=100 n_jobs=38 data_tag=tg
# make run_tuning_catboost input_file=../../data/clean/oxides_Tg_train.csv output_folder=../../result/ max_iter=100 seed=200 n_jobs=38 data_tag=tg
# make run_tuning_catboost input_file=../../data/clean/oxides_Tg_train.csv output_folder=../../result/ max_iter=100 seed=300 n_jobs=38 data_tag=tg
# make run_tuning_catboost input_file=../../data/clean/oxides_Tg_train.csv output_folder=../../result/ max_iter=100 seed=400 n_jobs=38 data_tag=tg
# make run_tuning_catboost input_file=../../data/clean/oxides_Tg_train.csv output_folder=../../result/ max_iter=100 seed=500 n_jobs=38 data_tag=tg
# ##########################################################################

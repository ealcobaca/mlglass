input_file := ../../data/clean/oxides_Tg_train.csv
output_folder := ../../result/
max_iter := 100
seed := 2019	
n_jobs := 5
data_tag := tg

run_clean_data:
	@printf "<<< Running .... >>>\n\n"
	@cd clean_data; python run.py
	@printf "\n\n<<< End .... >>>\n"
run_range:
	@printf "<<< Running .... >>>\n\n"
	@cd range; python run.py --n_cpus=8 -e 1
	@printf "\n\n<<< End .... >>>\n"
run_range_percentil:
	@printf "<<< Running .... >>>\n\n"
	@cd range; python run.py --n_cpus=8 -e 2
	@printf "\n\n<<< End .... >>>\n"
run_tuning_dt:
	@cd tuning; python hyperspaces_glass.py dt $(input_file) $(output_folder) $(max_iter) $(seed) $(n_jobs) $(data_tag)
run_tuning_rf:
	@cd tuning; python hyperspaces_glass.py rf $(input_file) $(output_folder) $(max_iter) $(seed) $(n_jobs) $(data_tag)
run_tuning_catboost:
	@cd tuning; python hyperspaces_glass.py catboost $(input_file) $(output_folder) $(max_iter) $(seed) $(n_jobs) $(data_tag)
run_tuning_mlp:
	@cd tuning; python hyperspaces_glass.py mlp $(input_file) $(output_folder) $(max_iter) $(seed) $(n_jobs) $(data_tag)
run_tuning_svr:
	@cd tuning; python hyperspaces_glass.py svr $(input_file) $(output_folder) $(max_iter) $(seed) $(n_jobs) $(data_tag)
run_tuning_knn:
	@cd tuning; python hyperspaces_glass.py knn $(input_file) $(output_folder) $(max_iter) $(seed) $(n_jobs) $(data_tag)

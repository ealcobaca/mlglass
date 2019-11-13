input_file := ../../data/other_properties/ICMC_oxides_Abbe.csv
output_folder := ../../all_properties_results/
max_iter := 500
n_jobs := 40
data_tag := abbe
outer_fold := 1

inner_seed := 0
outer_seed := 1
tuning_seed := 2

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
	@cd tuning; python hyperspaces_glass.py dt $(input_file) $(output_folder) $(max_iter) $(inner_seed) $(n_jobs) $(data_tag) $(outer_seed) $(tuning_seed) $(outer_fold)
run_tuning_rf:
	@cd tuning; python hyperspaces_glass.py rf $(input_file) $(output_folder) $(max_iter) $(inner_seed) $(n_jobs) $(data_tag) $(outer_seed) $(tuning_seed) $(outer_fold)
run_tuning_catboost:
	@cd tuning; python hyperspaces_glass.py catboost $(input_file) $(output_folder) $(max_iter) $(inner_seed) $(n_jobs) $(data_tag) $(outer_seed) $(tuning_seed) $(outer_fold)
run_tuning_mlp:
	@cd tuning; python hyperspaces_glass.py mlp $(input_file) $(output_folder) $(max_iter) $(inner_seed) $(n_jobs) $(data_tag) $(outer_seed) $(tuning_seed) $(outer_fold)
run_tuning_svr:
	@cd tuning; python hyperspaces_glass.py svr $(input_file) $(output_folder) $(max_iter) $(inner_seed) $(n_jobs) $(data_tag) $(outer_seed) $(tuning_seed) $(outer_fold)
run_tuning_knn:
	@cd tuning; python hyperspaces_glass.py knn $(input_file) $(output_folder) $(max_iter) $(inner_seed) $(n_jobs) $(data_tag) $(outer_seed) $(tuning_seed) $(outer_fold)
generate_logs:
	@cd tuning; python merge_tuning_results.py; python generate_best_models.py; python test_models.py; python log2tex.py
rf_vis:
	@cd tuning; python random_forest_visualization.py

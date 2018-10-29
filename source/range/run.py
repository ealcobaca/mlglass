from range_cutter import run
from range_cutter_percentil import run_percentil
from range_eval import run_eval
from range_eval_pbs import experiment_conf_id, run_eval_pbs
import sys, getopt, os

DATA_PATH = "../../data/clean/"

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "hn:e:i:",["n_cpus=","help=","exp="])
    except getopt.GetoptError:
        print('run.py [-n <n_cores>]')
        sys.exit(2)

    n_cores = 1
    aux=0
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print('run.py [-n <n_cores>]')
            sys.exit()
        elif opt in ("-n", "--n_cpus"):
            n_cores = int(arg)
        elif opt in ("-e", "--exp"):
            aux = int(arg)
        elif opt in ("-i"):
            id_exp = int(arg)

    if aux == 1:
        print("Experiment")
        run("../../data/clean/oxides_Tg_train.csv", "Tg", n_cores)
    elif aux == 2:
        print("Experiment with percentil")
        run_percentil("../../data/clean/oxides_Tg_train.csv", "Tg", n_cores)
    elif aux == 3:
        print("Experiment oracle")
        run_eval("../../data/clean/oxides_Tg_train.csv", "Tg", n_cores)
    elif aux == 4:
        print("Experiment oracle - PBS")
        result = experiment_conf_id("../../data/clean/oxides_Tg_train.csv", "Tg")
        run_eval_pbs(*result[id_exp])

        alg_path = "../../result/result_oracle/log/"
        if not os.path.exists(alg_path):
            os.makedirs(alg_path)
        file = open("{0}{1}.log".format(alg_path,id_exp), "w")
        file.close()

    else:
        sys.exit()

if __name__ == "__main__":
    main(sys.argv[1:])

from range_cutter import run
from range_cutter_percentil import run_percentil
import sys, getopt

DATA_PATH = "../../data/clean/"

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "hn:e:",["n_cpus=","help=","exp="])
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

    if aux == 1:
        print("Experiment")
        run("../../data/clean/oxides_Tg_train.csv", "Tg", n_cores)
    elif aux == 2:
        print("Experiment with percentil")
        run_percentil("../../data/clean/oxides_Tg_train.csv", "Tg", n_cores)
    else:
        sys.exit()

if __name__ == "__main__":
    main(sys.argv[1:])

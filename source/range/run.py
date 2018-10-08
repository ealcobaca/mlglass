from range_cutter import run
import sys, getopt

DATA_PATH = "../../data/clean/"

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "hn:",["n_cpus=","help="])
    except getopt.GetoptError:
        print('run.py [-n <n_cores>]')
        sys.exit(2)

    n_cores = 1
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print('run.py [-n <n_cores>]')
            sys.exit()
        elif opt in ("-n", "--n_cpus"):
            n_cores = int(arg)
            run("../../data/clean/oxides_Tg_train.csv", "Tg", n_cores)
        else:
            run("../../data/clean/oxides_Tg_train.csv", "Tg", n_cores)

if __name__ == "__main__":
    main(sys.argv[1:])

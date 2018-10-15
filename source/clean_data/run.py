import os
import pandas as pd
from data_cleaner import mult_target_split

data_path = "../../data/raw/"
save_path = "../../data/clean/"

def main():
    targets = ["Tg","ND300","Tliquidus"]
    data = pd.read_csv(data_path+'TgTliqND300oxides.csv', sep=',')
    data = data.drop(['Unnamed: 0'], axis=1)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    mult_target_split(data, targets, save_path, test_size=0.2, amount=6, seed=123)

if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd

data = pd.read_csv('../../data/raw/TgTliqND300oxides.csv')

# Rem
data = data[~pd.isna(data['Tg'])]

columns_to_drop = ['ND300', 'Tliquidus']
data = data.drop(columns=columns_to_drop)

dupl_idx = data.duplicated(subset=data.columns[1:-1], keep=False)
data_unique = data[~dupl_idx].copy()
data_duplicated = data[dupl_idx].copy()


unique_dupl_idx = data_duplicated.duplicated(
    subset=data_duplicated.columns[1:-1],
    keep='first'
)
unique_duplicated = data_duplicated[~unique_dupl_idx].copy()

for i in range(len(unique_duplicated)):
    print('{}/{} -- > {}'.format(i, len(unique_duplicated),
          len(data_duplicated)))

    aux = np.abs(
        data_duplicated.iloc[:, 1:-1] - unique_duplicated.iloc[i, 1:-1]
    )
    aux = np.isclose(aux.sum(axis=1), 0.0)
    aux = np.where(aux)[0].tolist()

    unique_duplicated.iloc[i, -1] = np.median(
        data_duplicated.iloc[aux, -1]
    ) if len(aux) > 1 else unique_duplicated.iloc[i, -1]
    data_duplicated.reset_index(drop=True, inplace=True)
    data_duplicated.drop(aux, inplace=True)

final = pd.concat([data_unique, unique_duplicated])
final.to_csv('../../data/clean/data_tg_names_dupl_rem.csv', index=False)
final.iloc[:, 1:].to_csv('../../data/clean/data_tg_dupl_rem.csv', index=False)

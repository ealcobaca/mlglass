import os


# DT
print('DT')
for i in range(1, 11):
    print('Fold {:02d}'.format(i))
    os.system('make run_tuning_dt n_jobs=2 outer_fold={0} tuning_seed={0}'.format(i))

# k-NN
print('\nk-NN')
for i in range(1, 11):
    print('Fold {:02d}'.format(i))
    os.system('make run_tuning_knn n_jobs=2 outer_fold={0} tuning_seed={0}'.format(i))

# RF
print('\nRF')
for i in range(1, 11):
    print('Fold {:02d}'.format(i))
    os.system('nohup make run_tuning_rf n_jobs=2 outer_fold={0} tuning_seed={0} > rf_{0:02d} &'.format(i))

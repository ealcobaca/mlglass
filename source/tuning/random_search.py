import numpy as np
from hp_space import HPSpace
from multiprocessing import Pool, Manager


class RandomSearch():
    """ Random Search method """

    def __init__(self, space, max_iter=10, n_jobs=1, random_state=None):
        """ Come thing
        Note:
            Do not include the `self` parameter in the ``Args`` section.
        Args:
            param1 (str): Description of `param1`.
            param2 (:obj:`int`, optional): Description of `param2`. Multiple
                lines are supported.
            param3 (:obj:`list` of :obj:`str`): Description of `param3`.
        """
        self.space = space
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        if random_state is not None:
            np.random.seed(random_state)

    def get_random_attr(self):
        conf = {}
        self.__get_random_attr(self.space, conf)

        return conf

    def __get_random_attr(self, space, conf):
        nro_branches = space.nro_branches()
        conf.update(space.get_data())

        if nro_branches:
            aux = np.random.randint(nro_branches)
            self.__get_random_attr(space.get_branch(aux), conf)


    def fmin(self, objective, **kwargs):
        if self.n_jobs > 1:
            return self.fmin_par(objective, **kwargs)
        return self.fmin_seq(objective, **kwargs)

    @classmethod
    def __takeFirst(cls, elem):
        return elem[0]

    @staticmethod
    def map_objective(param):
        value, objective, confs, kwargs = param
        conf = confs[value]
        aux = conf.copy()
        aux.update(kwargs)
        aux["id_tuning"] = str(value)
        value = objective(**aux)
        return value, conf

    def update_dic(self, a, b):
        a.update(b)
        return a

    def fmin_par(self, objective, **kwargs):
        with Pool(self.n_jobs) as pool:
            manager = Manager()
            self.objective = objective
            self.confs = [self.get_random_attr()
                          for i in range(0, self.max_iter)]

            confs_m = manager.list(self.confs)
            # kwargs_m = manager.list(kwargs)
            param = [(i, objective, confs_m, kwargs)
                     for i in range(0, self.max_iter)]
            result = pool.map(RandomSearch.map_objective, param)
            result.sort(key=self.__takeFirst)
            return result[0]

    def fmin_seq(self, objective, **kwargs):
        best_conf = self.get_random_attr()
        aux = best_conf.copy()
        aux.update(kwargs)
        best_value = objective(**aux)

        for t in range(1, self.max_iter):
            conf = self.get_random_attr()
            aux = conf.copy()
            aux.update(kwargs)
            value = objective(**aux)

            if value < best_value:
                best_value = value
                best_conf = conf

        return best_value, best_conf

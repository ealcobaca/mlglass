import numpy as np

class HPSpace(object):

    def __init__(self, name=None, parent=None):
        self.branches = []
        self.data = {}

        self.name = ""
        if name is not None:
            self.name = name

        if(parent is not None):
            parent.add_branch(self)


    def nro_branches(self):
        return len(self.branches)


    def add_branch(self, branch):
        self.branches.append(branch)

    def get_branch(self, idx):
        return self.branches[idx]


    def add_content(self, name, content):
        self.data[name] = content


    def get_value(self, k):
        c_type, s, e, f = self.data[k]
        value = None

        if c_type == 'c':
            value = f[np.random.randint(0,len(f))]
        elif c_type == 'z':
            value = int(np.rint(((e-s)*f()) + s))
        elif c_type == 'r':
            value = ((e-s)*f()) + s
        elif c_type == 'f':
            value = f()

        return value

    def get_data(self):
        return {k: self.get_value(k) for k in self.data.keys()}

    def get_content(self):
        return {k: self.get_value(**self.data[k]) for k in self.data.keys()}

    def print(self, space="  ", data=False):
        print("{0}|__ {1}".format(space, self.name))

        if data == True:
            for k in self.data.keys():
                print("{0}   {1}".format(space, k))

        for c in self.branches:
            c.print(space=space+"  ", data=data)

    @classmethod
    def add_axis(cls, branch, axis_name, axis_type, axis_min, axis_max,
                 axis_value):
        branch.add_content(axis_name, [axis_type, axis_min, axis_max,
                                       axis_value])

    @classmethod
    def new_branch(cls, father, name=None):
        return HPSpace(name=name, parent=father)

from abc import ABCMeta, abstractmethod
import numpy as np

class AbstractStructureDesigner(object, metaclass=ABCMeta):
    def __init__(self):
        self._nn_distance=None; self._kd_tree=None
    def _get_neighbours(self, query):
        if isinstance(query,(list,np.ndarray)):
            ans=self._kd_tree.query(query,k=25,distance_upper_bound=self._nn_distance)
        elif isinstance(query,int):
            query=list(self.atom_list.items())[query][1]
            if self._kd_tree.data.shape[1]>3:
                query=np.append(query,0)
            ans=self._kd_tree.query(query,k=25,distance_upper_bound=self._nn_distance)
        elif isinstance(query,str):
            ans=self._kd_tree.query(self.atom_list[query],k=25,distance_upper_bound=self._nn_distance)
        else:
            raise TypeError('Wrong input type for query')
        return ans
    @abstractmethod
    def get_neighbours(self, query):
        pass
    @property
    @abstractmethod
    def atom_list(self):
        pass

class AbstractBasis(object, metaclass=ABCMeta):
    @abstractmethod
    def qn2ind(self, qn):
        pass
    @abstractmethod
    def ind2qn(self, ind):
        pass
    @property
    @abstractmethod
    def orbitals_dict(self):
        pass

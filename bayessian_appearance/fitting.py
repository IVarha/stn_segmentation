

import point_distribution as pd





class Fitter:

    _pdm = None

    _train_subj = None
    _test_subj = None



    def __init__(self,train_subj,test_subj,pdm=None):
        self._train_subj = train_subj
        self._test_subj = test_subj
        if pdm != None:
            if isinstance(pdm,str):
                self._pdm = pd.PointDistribution.read_pdm(file_name=pdm)

            else:
                self._pdm = pdm

    def read_pdm(self, filename ):
        self._pdm = pd.PointDistribution.read_pdm(file_name=filename)

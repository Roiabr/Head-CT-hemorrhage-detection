"""import sys
import dataset as ds
from radio import CTImagesMaskedBatch

data_mask = '/head_ct/*.png'
"""
import pydicom as pdcm

fpath = 'complex_head_ct/CT000000.dcm'
fdataset = pdcm.dcmread(fpath)
print(fdataset)

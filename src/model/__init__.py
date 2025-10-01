# from model.bsarec import BSARecModel
# from model.caser import CaserModel
# from model.gru4rec import GRU4RecModel
# from model.sasrec import SASRecModel
# from model.bert4rec import BERT4RecModel
# from model.fmlprec import FMLPRecModel
# from model.duorec import DuoRecModel
# from model.fearec import FEARecModel
# from model.myfreqrec import MyFreqRecModel  # Assuming MyFreqRecModel is defined elsewhere
# from model.myfreqrec2 import MyFreqRec2Model  # Assuming MyFreqRec2Model is defined elsewhere
# from model.myfreqrec3 import MyFreqRec3Model  # Assuming MyFreqRec3Model is defined elsewhere
# from model.waverec import WaveRecModel  # Assuming WaveRecModel is defined elsewhere
# from model.mixedrec import MixedRecModel  # Assuming MixedRecModel is defined elsewhere
# from model.mixedbsarec import MixedBSARecModel  # Assuming MixedBSARecModel is defined elsewhere
# from model.cfit4srec import CFIT4SRecModel  # Assuming CFIT4SRecModel is defined elsewhere
# from model.sasrecwhite import SASRecSpectralWhiteningModel  # Assuming SASRecSpectralWhiteningModel is defined elsewhere
# from model.cl4srec import CL4SRecModel  # Assuming CL4SRecModel is defined elsewhere
# from model.meanrec import MeanRecModel  # Assuming MeanRecModel is defined elsewhere
from model.siclrec import SICLRecModel  # Assuming SICLRecModel is defined elsewhere


MODEL_DICT = {
    # "bsarec": BSARecModel,
    # "caser": CaserModel,
    # "gru4rec": GRU4RecModel,
    # "sasrec": SASRecModel,
    # "bert4rec": BERT4RecModel,
    # "fmlprec": FMLPRecModel,
    # "duorec": DuoRecModel,
    # "fearec": FEARecModel,
    # "cfit4srec":CFIT4SRecModel,  # Assuming CFIT4SRecModel is defined elsewhere
    # "cl4srec": CL4SRecModel,  # Assuming CL4SRecModel is defined elsewhere
    # "myfreqrec":MyFreqRecModel,  # Assuming MyRecModel is defined elsewhere
    # "myfreqrec2":MyFreqRec2Model,  # Assuming MyRecModel is defined elsewhere
    # "myfreqrec3":MyFreqRec3Model,  # Assuming MyRecModel is defined elsewhere
    # "waverec":WaveRecModel, # Assuming WaveRecModel is defined elsewhere
    # "mixedrec": MixedRecModel, # Assuming MixedRecModel is defined elsewhere
    # "mixedbsarec": MixedBSARecModel,  # Assuming BSARecModel is used for mixed BSARec
    # "sasrecspectralwhitening": SASRecSpectralWhiteningModel,  # Assuming SASRecModel is used for white noise filtering
    # "meanrec": MeanRecModel,  # Assuming MeanRecModel is defined elsewhere
    "siclrec": SICLRecModel,  # Assuming SICLRecModel is defined elsewhere

    }
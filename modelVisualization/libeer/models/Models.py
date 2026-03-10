from LibEER.models.DGCNN import DGCNN
# from models.RGNN import RGNN
from LibEER.models.RGNN_official import SymSimGCNNet
from LibEER.models.EEGNet import EEGNet
from LibEER.models.STRNN import STRNN
from LibEER.models.GCBNet import GCBNet
from LibEER.models.DBN import DBN
from LibEER.models.TSception import TSception
from LibEER.models.SVM import SVM
from LibEER.models.CDCN import CDCN
from LibEER.models.HSLT import HSLT
from LibEER.models.ACRNN import ACRNN
from LibEER.models.GCBNet_BLS import GCBNet_BLS
from LibEER.models.MsMda import MSMDA


Model = {
    'DGCNN': DGCNN,
    # 'RGNN': RGNN,
    'RGNN_official': SymSimGCNNet,
    'GCBNet': GCBNet,
    'GCBNet_BLS': GCBNet_BLS,
    'CDCN': CDCN,
    'DBN': DBN,
    'STRNN': STRNN,
    'EEGNet': EEGNet,
    'HSLT': HSLT,
    'ACRNN': ACRNN,
    'TSception': TSception,
    'MsMda': MSMDA,
    'svm' : SVM,
}

from LibEER import (EEGNet_train,TSception_train,GCBNet_train,
                    DGCNN_train,RGNN_train,GCBNet_BLS_train,
                    HSLT_train,DBN_train,CDCN_train,
                    Msmda_train,ACRNN_train,svm_train)

def ModelCenter(args):
    modelName = args.model
    print("训练的模型是",modelName)
    if modelName == "EEGNet":
        EEGNet_train.main(args)
    elif modelName == "TSception":
        TSception_train.main(args)
    elif modelName == "GCBNet":
        GCBNet_train.main(args)
    elif modelName == "DGCNN":
        DGCNN_train.main(args)
    elif modelName == "RGNN":
        RGNN_train.main(args)
    elif modelName == "GCBNet_BLS":
        GCBNet_BLS_train.main(args)
    elif modelName == "HSLT":
        HSLT_train.main(args)
    elif modelName == "DBN":
        DBN_train.main(args)
    elif modelName == "CDCN":
        CDCN_train.main(args)
    elif modelName == "Msmda":
        Msmda_train.main(args)
    elif modelName == "ACRNN":
        ACRNN_train.main(args)
    elif modelName == "svm":
        svm_train.main(args)
    else:
        print("Model name not recognized")
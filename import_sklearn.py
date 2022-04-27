# -*-coding:utf-8 -*-
'''
@File    :   import_sklearn.py
@Time    :   2022/02/28 16:47
@Author  :   Ryan Ma
@Version :   1.0
@Contact :   ryanma9629@gmail.com
@Desc    :   None
'''

import os
from pathlib import Path
import warnings
import pandas as pd
from sasctl import pzmm

warnings.filterwarnings("ignore", category=FutureWarning)


def import_sklearn(proj, model, name, desc, algorithm, modeler,
                   x_train, y_train, x_test, y_test, target_event,
                   model_folder, conn):

    Path(model_folder).mkdir(parents=True, exist_ok=True)

    files = os.listdir(model_folder)
    for f in files:
        if f.endswith(('.json', '.sas', '.py', '.pickle', '.zip')):
            os.remove(os.path.join(model_folder, f))

    # generate model pickle file
    pzmm.PickleModel().pickleTrainedModel(model, name, model_folder)

    # Write input variable mapping to a json file
    jf = pzmm.JSONFiles()
    jf.writeVarJSON(x_train, isInput=True, jPath=model_folder)

    # Set output variables and assign an event threshold, then write output variable mapping
    out_var = pd.DataFrame(
        columns=['EM_EVENTPROBABILITY', 'EM_CLASSIFICATION'])
    out_var['EM_CLASSIFICATION'] = y_train.astype('str').unique()
    out_var['EM_EVENTPROBABILITY'] = 0.5  # Event threshold
    jf.writeVarJSON(out_var, isInput=False, jPath=model_folder)

    # Write model properties to a json file
    jf.writeModelPropertiesJSON(modelName=name,
                                modelDesc=desc,
                                targetVariable=y_train.name,
                                modelType=algorithm,
                                modelPredictors=list(x_train.columns),
                                targetEvent=target_event,
                                numTargetCategories=len(y_train.unique()),
                                eventProbVar='EM_EVENTPROBABILITY',
                                jPath=model_folder,
                                modeler=modeler)
    # Write model metadata to a json file
    jf.writeFileMetadataJSON(name, jPath=model_folder)

    # Calculate train predictions
    train_proba = model.predict_proba(x_train)
    test_proba = model.predict_proba(x_test)

    # Assign data to lists of actual and predicted values
    train_res = pd.concat([y_train.reset_index(drop=True),
                           pd.Series(train_proba[:, 1])], axis=1)
    test_res = pd.concat([y_test.reset_index(drop=True),
                          pd.Series(test_proba[:, 1])], axis=1)

    # Calculate the model statistics and write to json files
    jf.calculateFitStat(trainData=train_res, testData=test_res,
                        jPath=model_folder)
    jf.generateROCLiftStat(y_train.name, target_event,
                           conn,
                           trainData=train_res, testData=test_res,
                           jPath=model_folder)

    pzmm.ImportModel().pzmmImportModel(model_folder, name, proj, x_train, y_train,
                                       '{}.predict_proba({})', force=True)

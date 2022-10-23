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


def import_sklearn(viya_conn,
                   mod_folder,
                   proj,
                   model,
                   mod_name,
                   mod_desc,
                   mod_algo,
                   mod_owner,
                   target_event,
                   X_train,
                   y_train,
                   X_test=None,
                   y_test=None):

    Path(mod_folder).mkdir(parents=True, exist_ok=True)

    files = os.listdir(mod_folder)
    for f in files:
        if f.endswith(('.json', '.sas', '.py', '.pickle', '.zip')):
            os.remove(os.path.join(mod_folder, f))

    # generate model pickle file
    pzmm.PickleModel().pickleTrainedModel(model, mod_name, mod_folder)

    # Write input variable mapping to a json file
    jf = pzmm.JSONFiles()
    jf.writeVarJSON(X_train, isInput=True, jPath=mod_folder)

    # Set output variables and assign an event threshold, then write output variable mapping
    out_var = pd.DataFrame(
        columns=['EM_EVENTPROBABILITY', 'EM_CLASSIFICATION'])
    out_var['EM_CLASSIFICATION'] = y_train.astype('str').unique()
    out_var['EM_EVENTPROBABILITY'] = 0.5  # Event threshold
    jf.writeVarJSON(out_var, isInput=False, jPath=mod_folder)

    # Write model properties to a json file
    jf.writeModelPropertiesJSON(modelName=mod_name,
                                modelDesc=mod_desc,
                                targetVariable=y_train.name,
                                modelType=mod_algo,
                                modeler=mod_owner,
                                modelPredictors=list(X_train.columns),
                                targetEvent=target_event,
                                numTargetCategories=len(y_train.unique()),
                                eventProbVar='EM_EVENTPROBABILITY',
                                jPath=mod_folder)
    # Write model metadata to a json file
    jf.writeFileMetadataJSON(mod_name, jPath=mod_folder)

    # Calculate train predictions
    if X_test is not None and y_test is not None:
        train_proba = model.predict_proba(X_train)
        test_proba = model.predict_proba(X_test)
        train_res = pd.concat([y_train.reset_index(drop=True),
                               pd.Series(train_proba[:, 1])], axis=1)
        test_res = pd.concat([y_test.reset_index(drop=True),
                              pd.Series(test_proba[:, 1])], axis=1)

        # Calculate the model statistics and write to json files
        jf.calculateFitStat(trainData=train_res, testData=test_res,
                            jPath=mod_folder)
        jf.generateROCLiftStat(y_train.name, target_event,
                               viya_conn,
                               trainData=train_res, testData=test_res,
                               jPath=mod_folder)

    pzmm.ImportModel().pzmmImportModel(mod_folder, mod_name, proj, X_train, y_train,
                                       '{}.predict_proba({})', force=True)

import math
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

import settings

with open(Path(settings.pickle_path) / "DecisionTree.pickle", "rb") as pickle_model:
    model = pickle.load(pickle_model)

def score(LOAN, MORTDUE, VALUE, REASON, JOB, YOJ, DEROG, DELINQ, CLAGE, NINQ, CLNO, DEBTINC):
    "Output: EM_CLASSIFICATION, EM_EVENTPROBABILITY"

    try:
        global model
    except NameError:
        with open(Path(settings.pickle_path) / "DecisionTree.pickle", "rb") as pickle_model:
                model = pickle.load(pickle_model)

    input_array = pd.DataFrame([[LOAN, MORTDUE, VALUE, REASON, JOB, YOJ, DEROG, DELINQ, CLAGE, NINQ, CLNO, DEBTINC]],
                              columns=["LOAN", "MORTDUE", "VALUE", "REASON", "JOB", "YOJ", "DEROG", "DELINQ", "CLAGE", "NINQ", "CLNO", "DEBTINC"],
                              dtype=float)
    prediction = model.predict_proba(input_array)

    # Check for numpy values and convert to a CAS readable representation
    if isinstance(prediction, np.ndarray):
        prediction = prediction.tolist()[0]

    if prediction[0] > prediction[1]:
        EM_CLASSIFICATION = "0"
    else:
        EM_CLASSIFICATION = "1"

    return EM_CLASSIFICATION, prediction[0]
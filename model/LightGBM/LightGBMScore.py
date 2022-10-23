

import math
import pickle
import pandas as pd
import numpy as np

with open('/models/resources/viya/be2d1628-0a43-4a50-9b7e-d9d09f552d53/LightGBM.pickle', 'rb') as _pFile:
    _thisModelFit = pickle.load(_pFile)

def scoreLightGBM(A, B, C):
    "Output: EM_EVENTPROBABILITY, EM_CLASSIFICATION"

    try:
        global _thisModelFit
    except NameError:

        with open('/models/resources/viya/be2d1628-0a43-4a50-9b7e-d9d09f552d53/LightGBM.pickle', 'rb') as _pFile:
            _thisModelFit = pickle.load(_pFile)

    try:
        inputArray = pd.DataFrame([[A, B, C]],
                                  columns=['A', 'B', 'C'],
                                  dtype=float)
        prediction = _thisModelFit.predict_proba(inputArray)
    except ValueError:
    # For models requiring or including an intercept value, a 'const' column is required
    # For example, many statsmodels models include an intercept value that must be included for the model prediction
        inputArray = pd.DataFrame([[1.0, A, B, C]],
                                columns=['const', 'A', 'B', 'C'],
                                dtype=float)
        prediction = _thisModelFit.predict_proba(inputArray)

    try:
        EM_EVENTPROBABILITY = float(prediction)
    except TypeError:
    # If the model expects non-binary responses, a TypeError will be raised.
    # The except block shifts the prediction to accept a non-binary response.
        EM_EVENTPROBABILITY = float(prediction[:,1])

    if (EM_EVENTPROBABILITY >= 0.5428571428571428):
        EM_CLASSIFICATION = '1'
    else:
        EM_CLASSIFICATION = '0' 

    return(EM_EVENTPROBABILITY, EM_CLASSIFICATION)



import math
import pickle
import pandas as pd
import numpy as np

with open('/models/resources/viya/df26e784-5dca-4fb6-9d74-37b5e0528cc4/LR.pickle', 'rb') as _pFile:
    _thisModelFit = pickle.load(_pFile)

def scoreLR(mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness, mean_compactness, mean_concavity, mean_concave_points, mean_symmetry, mean_fractal_dimension, radius_error, texture_error, perimeter_error, area_error, smoothness_error, compactness_error, concavity_error, concave_points_error, symmetry_error, fractal_dimension_error, worst_radius, worst_texture, worst_perimeter, worst_area, worst_smoothness, worst_compactness, worst_concavity, worst_concave_points, worst_symmetry, worst_fractal_dimension):
    "Output: EM_EVENTPROBABILITY, EM_CLASSIFICATION"

    try:
        global _thisModelFit
    except NameError:

        with open('/models/resources/viya/df26e784-5dca-4fb6-9d74-37b5e0528cc4/LR.pickle', 'rb') as _pFile:
            _thisModelFit = pickle.load(_pFile)

    try:
        inputArray = pd.DataFrame([[mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness, mean_compactness, mean_concavity, mean_concave_points, mean_symmetry, mean_fractal_dimension, radius_error, texture_error, perimeter_error, area_error, smoothness_error, compactness_error, concavity_error, concave_points_error, symmetry_error, fractal_dimension_error, worst_radius, worst_texture, worst_perimeter, worst_area, worst_smoothness, worst_compactness, worst_concavity, worst_concave_points, worst_symmetry, worst_fractal_dimension]],
                                  columns=['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness', 'mean_compactness', 'mean_concavity', 'mean_concave_points', 'mean_symmetry', 'mean_fractal_dimension', 'radius_error', 'texture_error', 'perimeter_error', 'area_error', 'smoothness_error', 'compactness_error', 'concavity_error', 'concave_points_error', 'symmetry_error', 'fractal_dimension_error', 'worst_radius', 'worst_texture', 'worst_perimeter', 'worst_area', 'worst_smoothness', 'worst_compactness', 'worst_concavity', 'worst_concave_points', 'worst_symmetry', 'worst_fractal_dimension'],
                                  dtype=float)
        prediction = _thisModelFit.predict_proba(inputArray)
    except ValueError:
    # For models requiring or including an intercept value, a 'const' column is required
    # For example, many statsmodels models include an intercept value that must be included for the model prediction
        inputArray = pd.DataFrame([[1.0, mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness, mean_compactness, mean_concavity, mean_concave_points, mean_symmetry, mean_fractal_dimension, radius_error, texture_error, perimeter_error, area_error, smoothness_error, compactness_error, concavity_error, concave_points_error, symmetry_error, fractal_dimension_error, worst_radius, worst_texture, worst_perimeter, worst_area, worst_smoothness, worst_compactness, worst_concavity, worst_concave_points, worst_symmetry, worst_fractal_dimension]],
                                columns=['const', 'mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness', 'mean_compactness', 'mean_concavity', 'mean_concave_points', 'mean_symmetry', 'mean_fractal_dimension', 'radius_error', 'texture_error', 'perimeter_error', 'area_error', 'smoothness_error', 'compactness_error', 'concavity_error', 'concave_points_error', 'symmetry_error', 'fractal_dimension_error', 'worst_radius', 'worst_texture', 'worst_perimeter', 'worst_area', 'worst_smoothness', 'worst_compactness', 'worst_concavity', 'worst_concave_points', 'worst_symmetry', 'worst_fractal_dimension'],
                                dtype=float)
        prediction = _thisModelFit.predict_proba(inputArray)

    try:
        EM_EVENTPROBABILITY = float(prediction)
    except TypeError:
    # If the model expects non-binary responses, a TypeError will be raised.
    # The except block shifts the prediction to accept a non-binary response.
        EM_EVENTPROBABILITY = float(prediction[:,1])

    if (EM_EVENTPROBABILITY >= 0.6381909547738693):
        EM_CLASSIFICATION = '1'
    else:
        EM_CLASSIFICATION = '0' 

    return(EM_EVENTPROBABILITY, EM_CLASSIFICATION)

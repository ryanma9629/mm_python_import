data sasep.out;
dcl package pymas pm;
dcl package logger logr('App.tk.MAS');
dcl varchar(32767) character set utf8 pypgm;
dcl double resultCode revision;
dcl varchar(100) EM_CLASSIFICATION;
dcl double EM_EVENTPROBABILITY;


method score(double mean_radius, double mean_texture, double mean_perimeter, double mean_area, double mean_smoothness, double mean_compactness, double mean_concavity, double mean_concave_points, double mean_symmetry, double mean_fractal_dimension, double radius_error, double texture_error, double perimeter_error, double area_error, double smoothness_error, double compactness_error, double concavity_error, double concave_points_error, double symmetry_error, double fractal_dimension_error, double worst_radius, double worst_texture, double worst_perimeter, double worst_area, double worst_smoothness, double worst_compactness, double worst_concavity, double worst_concave_points, double worst_symmetry, double worst_fractal_dimension, in_out double resultCode, in_out double EM_EVENTPROBABILITY, in_out varchar(100) EM_CLASSIFICATION);
   resultCode = revision = 0;
   if null(pm) then do;
      pm = _new_ pymas();
      resultCode = pm.useModule('model_exec_14354de1-b366-4609-815b-de0d55bfa090', 1);
      if resultCode then do;
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('import math');
         resultCode = pm.appendSrcLine('import pickle');
         resultCode = pm.appendSrcLine('import pandas as pd');
         resultCode = pm.appendSrcLine('import numpy as np');
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('with open("/models/resources/viya/df26e784-5dca-4fb6-9d74-37b5e0528cc4/LR.pickle", "rb") as _pFile:');
         resultCode = pm.appendSrcLine('    _thisModelFit = pickle.load(_pFile)');
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('def scoreLR(mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness, mean_compactness, mean_concavity, mean_concave_points, mean_symmetry, mean_fractal_dimension, radius_error, texture_error, perimeter_error, area_error, smoothness_error, compactness_error, concavity_error, concave_points_error, symmetry_error, fractal_dimension_error, worst_radius, worst_texture, worst_perimeter, worst_area, worst_smoothness, worst_compactness, worst_concavity, worst_concave_points, worst_symmetry, worst_fractal_dimension):');
         resultCode = pm.appendSrcLine('    "Output: EM_EVENTPROBABILITY, EM_CLASSIFICATION"');
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('    try:');
         resultCode = pm.appendSrcLine('        global _thisModelFit');
         resultCode = pm.appendSrcLine('    except NameError:');
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('        with open("/models/resources/viya/df26e784-5dca-4fb6-9d74-37b5e0528cc4/LR.pickle", "rb") as _pFile:');
         resultCode = pm.appendSrcLine('            _thisModelFit = pickle.load(_pFile)');
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('    try:');
         resultCode = pm.appendSrcLine('        inputArray = pd.DataFrame([[mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness, mean_compactness, mean_concavity, mean_concave_points, mean_symmetry, mean_fractal_dimension, radius_error, texture_error, perimeter_error, area_error, smoothness_error, compactness_error, concavity_error, concave_points_error, symmetry_error, fractal_dimension_error, worst_radius, worst_texture, worst_perimeter, worst_area, worst_smoothness, worst_compactness, worst_concavity, worst_concave_points, worst_symmetry, worst_fractal_dimension]],');
         resultCode = pm.appendSrcLine('                                  columns=["mean_radius", "mean_texture", "mean_perimeter", "mean_area", "mean_smoothness", "mean_compactness", "mean_concavity", "mean_concave_points", "mean_symmetry", "mean_fractal_dimension", "radius_error", "texture_error", "perimeter_error", "area_error", "smoothness_error", "compactness_error", "concavity_error", "concave_points_error", "symmetry_error", "fractal_dimension_error", "worst_radius", "worst_texture", "worst_perimeter", "worst_area", "worst_smoothness", "worst_compactness", "worst_concavity", "worst_concave_points", "worst_symmetry", "worst_fractal_dimension"],');
         resultCode = pm.appendSrcLine('                                  dtype=float)');
         resultCode = pm.appendSrcLine('        prediction = _thisModelFit.predict_proba(inputArray)');
         resultCode = pm.appendSrcLine('    except ValueError:');
         resultCode = pm.appendSrcLine('    # For models requiring or including an intercept value, a "const" column is required');
         resultCode = pm.appendSrcLine('    # For example, many statsmodels models include an intercept value that must be included for the model prediction');
         resultCode = pm.appendSrcLine('        inputArray = pd.DataFrame([[1.0, mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness, mean_compactness, mean_concavity, mean_concave_points, mean_symmetry, mean_fractal_dimension, radius_error, texture_error, perimeter_error, area_error, smoothness_error, compactness_error, concavity_error, concave_points_error, symmetry_error, fractal_dimension_error, worst_radius, worst_texture, worst_perimeter, worst_area, worst_smoothness, worst_compactness, worst_concavity, worst_concave_points, worst_symmetry, worst_fractal_dimension]],');
         resultCode = pm.appendSrcLine('                                columns=["const", "mean_radius", "mean_texture", "mean_perimeter", "mean_area", "mean_smoothness", "mean_compactness", "mean_concavity", "mean_concave_points", "mean_symmetry", "mean_fractal_dimension", "radius_error", "texture_error", "perimeter_error", "area_error", "smoothness_error", "compactness_error", "concavity_error", "concave_points_error", "symmetry_error", "fractal_dimension_error", "worst_radius", "worst_texture", "worst_perimeter", "worst_area", "worst_smoothness", "worst_compactness", "worst_concavity", "worst_concave_points", "worst_symmetry", "worst_fractal_dimension"],');
         resultCode = pm.appendSrcLine('                                dtype=float)');
         resultCode = pm.appendSrcLine('        prediction = _thisModelFit.predict_proba(inputArray)');
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('    try:');
         resultCode = pm.appendSrcLine('        EM_EVENTPROBABILITY = float(prediction)');
         resultCode = pm.appendSrcLine('    except TypeError:');
         resultCode = pm.appendSrcLine('    # If the model expects non-binary responses, a TypeError will be raised.');
         resultCode = pm.appendSrcLine('    # The except block shifts the prediction to accept a non-binary response.');
         resultCode = pm.appendSrcLine('        EM_EVENTPROBABILITY = float(prediction[:,1])');
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('    if (EM_EVENTPROBABILITY >= 0.6381909547738693):');
         resultCode = pm.appendSrcLine('        EM_CLASSIFICATION = "1"');
         resultCode = pm.appendSrcLine('    else:');
         resultCode = pm.appendSrcLine('        EM_CLASSIFICATION = "0" ');
         resultCode = pm.appendSrcLine('');
         resultCode = pm.appendSrcLine('    return(EM_EVENTPROBABILITY, EM_CLASSIFICATION)');
         revision = pm.publish(pm.getSource(), 'model_exec_14354de1-b366-4609-815b-de0d55bfa090');

         if ( revision < 1 ) then do;
            logr.log( 'e', 'py.publish() failed.');
            resultCode = -1;
            return;
         end;
      end;
   end;
   resultCode = pm.useMethod('scoreLR');
   if resultCode then do;
      logr.log('E', 'useMethod() failed. resultCode=$s', resultCode);
      return;
   end;
   resultCode = pm.setDouble('mean_radius', mean_radius);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('mean_texture', mean_texture);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('mean_perimeter', mean_perimeter);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('mean_area', mean_area);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('mean_smoothness', mean_smoothness);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('mean_compactness', mean_compactness);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('mean_concavity', mean_concavity);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('mean_concave_points', mean_concave_points);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('mean_symmetry', mean_symmetry);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('mean_fractal_dimension', mean_fractal_dimension);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('radius_error', radius_error);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('texture_error', texture_error);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('perimeter_error', perimeter_error);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('area_error', area_error);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('smoothness_error', smoothness_error);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('compactness_error', compactness_error);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('concavity_error', concavity_error);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('concave_points_error', concave_points_error);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('symmetry_error', symmetry_error);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('fractal_dimension_error', fractal_dimension_error);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('worst_radius', worst_radius);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('worst_texture', worst_texture);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('worst_perimeter', worst_perimeter);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('worst_area', worst_area);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('worst_smoothness', worst_smoothness);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('worst_compactness', worst_compactness);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('worst_concavity', worst_concavity);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('worst_concave_points', worst_concave_points);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('worst_symmetry', worst_symmetry);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.setDouble('worst_fractal_dimension', worst_fractal_dimension);
   if resultCode then
      logr.log('E', 'useMethod() failed.  resultCode=$s', resultCode);
   resultCode = pm.execute();
   if (resultCode) then put 'Error: pm.execute failed.  resultCode=' resultCode;
   else do;
      EM_EVENTPROBABILITY = pm.getDouble('EM_EVENTPROBABILITY');
      EM_CLASSIFICATION = pm.getString('EM_CLASSIFICATION');
   end;
end;

method run();
    set SASEP.IN;
    score(mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness, mean_compactness, mean_concavity, mean_concave_points, mean_symmetry, mean_fractal_dimension, radius_error, texture_error, perimeter_error, area_error, smoothness_error, compactness_error, concavity_error, concave_points_error, symmetry_error, fractal_dimension_error, worst_radius, worst_texture, worst_perimeter, worst_area, worst_smoothness, worst_compactness, worst_concavity, worst_concave_points, worst_symmetry, worst_fractal_dimension, resultCode, EM_EVENTPROBABILITY, EM_CLASSIFICATION);
end;
enddata;


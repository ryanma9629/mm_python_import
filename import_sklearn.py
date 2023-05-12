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
import pandas as pd
from sasctl import pzmm


def import_sklearn(project_name,
                   model_object,
                   model_name,
                   model_description,
                   model_algorithm,
                   model_owner,
                   target_event,
                   X_train,
                   y_train,
                   model_folder,
                   X_test=None,
                   y_test=None):

    Path(model_folder).mkdir(parents=True, exist_ok=True)

    files = os.listdir(model_folder)
    for f in files:
        if f.endswith(('.json', '.sas', '.py', '.pickle', '.zip')):
            os.remove(os.path.join(model_folder, f))

    # generate model pickle file
    pzmm.PickleModel().pickle_trained_model(model_prefix=model_name,
                                            trained_model=model_object,
                                            pickle_path=model_folder,
                                            is_h2o_model=False)

    # Write input variable mapping to a json file
    pzmm.JSONFiles.write_var_json(input_data=X_train,
                                  is_input=True,
                                  json_path=model_folder)

    # Set output variables and assign an event threshold, then write output variable mapping
    score_metrics = ["EM_CLASSIFICATION", "EM_EVENTPROBABILITY"]
    output_df = pd.DataFrame(columns=score_metrics)
    output_df[score_metrics[0]] = y_train.astype('str').unique()
    output_df[score_metrics[1]] = 0.5  # Event threshold
    pzmm.JSONFiles.write_var_json(input_data=output_df,
                                  is_input=False,
                                  json_path=model_folder)

    # Write model properties to a json file
    pzmm.JSONFiles.write_model_properties_json(model_name=model_name,
                                               target_variable=y_train.name,
                                               target_values=list(
                                                   y_train.unique()),
                                               json_path=model_folder,
                                               model_desc=model_description,
                                               model_algorithm=model_algorithm,
                                               modeler=model_owner)

    # Write model metadata to a json file
    pzmm.JSONFiles.write_file_metadata_json(model_prefix=model_name,
                                            json_path=model_folder,
                                            is_h2o_model=False)

    # Calculate train predictions
    if X_test is not None and y_test is not None:
        train_proba = model_object.predict_proba(X_train)
        test_proba = model_object.predict_proba(X_test)
        train_res = pd.concat([y_train.reset_index(drop=True),
                               pd.Series(train_proba[:, 1])], axis=1)
        test_res = pd.concat([y_test.reset_index(drop=True),
                              pd.Series(test_proba[:, 1])], axis=1)

        # Calculate the model statistics and write to json files
        pzmm.JSONFiles.calculate_model_statistics(target_value=target_event,
                                                  prob_value=0.5,
                                                  train_data=train_res,
                                                  test_data=test_res,
                                                  json_path=model_folder)

    pzmm.ImportModel.import_model(model_files=model_folder,
                                  model_prefix=model_name,
                                  project=project_name,
                                  input_data=X_train,
                                  target_values=list(y_train.unique()),
                                  score_metrics=score_metrics,
                                  model_file_name=model_name + '.pickle',
                                  predict_method=[
                                      model_object.predict_proba, [float, float]],
                                  overwrite_model=True)


if __name__ == '__main__':
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sasctl import Session

    raw = pd.read_csv('data/hmeq.csv')
    col_y = 'BAD'
    col_X = raw.drop(col_y, axis=1).columns
    X = raw[col_X]
    y = raw[col_y]

    col_cat = X.columns[X.dtypes == 'O']
    col_num = X.columns[X.dtypes != 'O']
    X.loc[:, col_cat] = X[col_cat].fillna('X')
    X.loc[:, col_num] = X[col_num].fillna(0)
    le = LabelEncoder()
    for c in col_cat:
        X.loc[:, c] = le.fit_transform(X[c])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=123)

    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    viya_user = 'sasdemo1'
    viya_pwd = 'Orion123'
    viya_host = 'viya4'
    viya_session = Session(viya_host, viya_user, viya_pwd, protocol='http')
    viya_connection = viya_session.as_swat()

    model_objects = [dt, lr, rf]
    model_names = ['DecisionTree', 'Logistic', 'RandomForest']
    model_descriptions = ['Description for the ' +
                          m + ' model' for m in model_names]
    model_algorithms = ['Decision Tree Classifier',
                        'Logistic Regression', 'Random Forest Classifier']
    model_folders = ['model/' + m for m in model_names]
    model_owner = 'Ryan Ma'
    target_event = 1
    project_name = 'HMEQ(Python) v202305121541'

    for (model_object, model_name, model_description, model_algorithm, model_folder) in zip(model_objects, model_names, model_descriptions, model_algorithms, model_folders):
        import_sklearn(project_name=project_name,
                       model_object=model_object,
                       model_name=model_name,
                       model_description=model_description,
                       model_algorithm=model_algorithm,
                       model_owner=model_owner,
                       target_event=target_event,
                       X_train=X_train,
                       y_train=y_train,
                       model_folder=model_folder,
                       X_test=X_test,
                       y_test=y_test)

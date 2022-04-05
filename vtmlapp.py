import streamlit as st
from joblib import load
import pandas as pd
import numpy as np

@st.cache
def import_data_andmodel():
    bdb = pd.read_excel('best_subdf.xlsx', index_col=0)
    nn_model = load('best_model.joblib')

    Xs = bdb.drop('recurrence_1y', axis=1).values
    ys = bdb.recurrence_1y.values

    n =200
    testX = Xs[n, :]
    prediction = nn_model.predict_proba([testX])
    return bdb, nn_model, Xs, ys


# print(prediction, ys[n])


def show_predict():
    st.title('VT 1 year recurrence prediction')
    st.write('''### first please input the patient data below: ''')

    bdb, nn_model, Xs, ys = import_data_andmodel()
    bdb_data_columns = bdb.drop('recurrence_1y', axis=1).columns

    patient_data = {}
    for colname in bdb_data_columns:
        values = bdb[colname].values
        value_number = len(np.unique(values))
        min_value = int(values.min())
        max_value = int(values.max())
        mean_values = int(values.mean())

        if value_number > 2:

            # print(colname, min_value, max_value)
            step = np.ones(1, 'int64')
            # print(step[0], type(step[0]), type(min_value))
            patient_data[colname] = st.slider(colname, min_value, max_value,  mean_values, 1)

        else:
            patient_data[colname] = int(st.radio(colname, [0, 1]))




    patient_data_forprediction = []

    for col in bdb_data_columns:
        patient_data_forprediction.append(patient_data[col])

    prediction = nn_model.predict_proba([patient_data_forprediction])

    st.metric(f'The prediction of the model for 1 year recurrence is:', prediction.max())

show_predict()
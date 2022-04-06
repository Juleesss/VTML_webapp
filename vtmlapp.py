import streamlit
import joblib
import numpy

@streamlit.cache
def import_data_andmodel():
    
    nn_model = joblib.load('best_model.joblib')
    columns = [
        'HTN', 'COPD', 'CRT', 'MI_base_x', 'MI34_base', 'LV_aneurysm_x', \
        'Furosemid', 'ARB', 'ICD_shock', 'Incessant_VT', 'Halmozott_ICD_therapy', \
        'Indukalhatosag', 'Indukalhato_morfologiak_szama', 'LAVA_abl']
    bdb = {
        'HTN' : [0, 1, 1], 'COPD' : [0, 1, 0], 'CRT' : [0, 1, 0], 'MI_base_x' : [0, 4, 2], 'MI34_base' : [0, 1, 1], 'LV_aneurysm_x' : [0, 1, 0], \
        'Furosemid' : [0, 1, 1], 'ARB' : [0, 0, 1], 'ICD_shock' : [0, 1, 1], 'Incessant_VT' : [0, 1, 0], 'Halmozott_ICD_therapy' : [0, 1, 0], \
        'Indukalhatosag' : [0, 1, 1], 'Indukalhato_morfologiak_szama' : [0, 5, 0], 'LAVA_abl' : [0, 1, 1]
        }
    return nn_model, columns, bdb


# print(prediction, ys[n])


def show_predict():
    streamlit.title('VT 1 year recurrence prediction')
    streamlit.write('''### first please input the patient data below: ''')

    nn_model, columns, bdb = import_data_andmodel()
    bdb_data_columns = columns

    patient_data = {}
    for colname in bdb_data_columns:
        values = numpy.array(bdb[colname])
        value_number = len(numpy.unique(values))
        min_value = int(values.min())
        max_value = int(values.max())
        mean_values = int(values.mean())

        if value_number > 2:
            patient_data[colname] = streamlit.slider(colname, min_value, max_value,  mean_values, 1)

        else:
            patient_data[colname] = int(streamlit.radio(colname, [0, 1]))




    patient_data_forprediction = []

    for col in bdb_data_columns:
        patient_data_forprediction.append(patient_data[col])

    prediction = nn_model.predict_proba([patient_data_forprediction])

    streamlit.metric(f'The prediction of the model for 1 year recurrence is:', prediction.max())

show_predict()

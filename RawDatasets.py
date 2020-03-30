
import pandas as pd
import json


def Process_Raw_Inputs():
    patients = []
    for patient_index, pat_data in df.groupby(['IDX', 'IDX_TYPE']):
        visits = []
        for visit_index, visit_data in pat_data.groupby(['BILLABLE_START_DT']):
            inputs = {}
            for idx, row in visit_data.iterrows():
                inputs[row['CODE_TYPE']] = str(row['CODE_CSV']).split(',')
            inputs['DAYS_SINCE_FIRST'] = row['DAYS_SINCE_FIRST']
            inputs['DAYS_SINCE_LAST'] = row['DAYS_SINCE_LAST']
            visits.append(inputs)
        patients.append(visits)
    return patients
   
def flatten_visit(visit, code_types):       
        codes = []
        for code_type in code_types:
            if code_type in visit.keys():
                codes.extend(visit[code_type])
        return codes

def Combine_Codes(patient_visit_list, code_types):
    """Combines codes within a visit based on the specified code_types.
    Parameters
    ----------
    patient_visit_list : list, required
        a list of of listed dictionaries. Patients and their visits and various code types 
        within that visit. [[{}, {}], [{}, {}, {}]] 
    Returns
    ------
    list of listed lists. Patients and their visits' combined codes
    """
    patients = []
    for patient in patient_visit_list:
        converted_visits = []
        for visit in patient:
            converted_visits.append(flatten_visit(visit, code_types))
        patients.append(converted_visits)
    return patients

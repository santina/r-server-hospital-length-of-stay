import pyodbc
from scipy.ndimage import imread
from revoscalepy import RxSqlServerData, rx_import, RxOdbcData, rx_write_object, rx_serialize_model, rx_read_object
from collections import OrderedDict
import numpy as np
from pandas import Series, DataFrame, to_numeric
import revoscalepy as rp
import sys
from math import sqrt

def make_train_test_data(input_sql, split_percentage, connection_string, id_colname, table_name, train_table_name):
    ## Create the Train_Id table containing Lead_Id of training set.
    train_test_split(id_colname, table_name, train_table_name, split_percentage, connection_string)
    col_type_and_factor_info = get_los_col_type_factor_info()

    ## Point to the training set. It will be created on the fly when training models.
    variables_all = rp.rx_get_var_names(input_sql)
    variables_to_remove = ["eid", "vdate", "discharged", "facid"]
    training_variables = [x for x in variables_all if x not in variables_to_remove]
    LoS_Train = rp.RxSqlServerData(sql_query = "SELECT eid, {} FROM LoS WHERE eid IN (SELECT eid from Train_Id)".format(
        ', '.join(training_variables)), connection_string = connection_string, column_info = col_type_and_factor_info
    )

    ## Point to the testing set. It will be created on the fly when testing models.
    LoS_Test = rp.RxSqlServerData(sql_query = "SELECT eid, {} FROM LoS WHERE eid NOT IN (SELECT eid from Train_Id)".format(
        ', '.join(training_variables)), connection_string = connection_string, column_info = col_type_and_factor_info
    )

    return LoS_Train, LoS_Test


def get_los_col_type_factor_info():
    col_type_and_factor_info = {"irondef": {"type": "factor", "levels":["0", "1"]},
                               "psychother": {"type": "factor", "levels":["0", "1"]},
                               "pulse": {"type": "numeric"},
                               "malnutrition": {"type": "factor", "levels":["0", "1"]},
                               "pneum": {"type": "factor", "levels":["0", "1"]},
                               "respiration": {"type": "numeric"},
                               "eid": {"type": "integer"},
                               "hematocrit": {"type": "numeric"},
                               "sodium": {"type": "numeric"},
                               "psychologicaldisordermajor": {"type": "factor", "levels":["0", "1"]},
                               "hemo": {"type": "factor", "levels":["0", "1"]},
                               "dialysisrenalendstage": {"type": "factor", "levels":["0", "1"]},
                               "discharged": {"type": "factor"},
                               "facid": {"type": "factor", "levels":["B", "A", "E", "D", "C"]},
                               "rcount": {"type": "factor", "levels":["0", "5+", "1", "4", "2", "3"]},
                               "substancedependence": {"type": "factor", "levels":["0", "1"]},
                               "number_of_issues": {"type": "factor", "levels":["0", "2", "1", "3", "4", "5", "6", "7", "8", "9"]},
                               "bmi": {"type": "numeric"},
                               "secondarydiagnosisnonicd9": {"type": "factor", "levels":["4", "1", "2", "3", "0", "7", "6", "10", "8", "5", "9"]},
                               "glucose": {"type": "numeric"},
                               "vdate": {"type": "factor"},
                               "asthma": {"type": "factor", "levels":["0", "1"]},
                               "depress": {"type": "factor", "levels":["0", "1"]},
                               "gender": {"type": "factor", "levels":["F", "M"]},
                               "fibrosisandother": {"type": "factor", "levels":["0", "1"]},
                               "lengthofstay": {"type": "numeric"},
                               "neutrophils": {"type": "numeric"},
                               "bloodureanitro": {"type": "numeric"},
                               "creatinine": {"type": "numeric"}}
    return col_type_and_factor_info


def get_los_col_info():
    col_type_info = {"eid": {'type': 'integer'},
                "rcount": {'type': 'character'},
                "vdate": {'type': 'character'},
                "gender": {'type': 'factor'},
                "dialysisrenalendstage": {'type': 'factor'},
                "asthma": {'type': 'factor'},
                "irondef": {'type': 'factor'},
                "pneum": {'type': 'factor'},
                "substancedependence": {'type': 'factor'},
                "psychologicaldisordermajor": {'type': 'factor'},
                "depress": {'type': 'factor'},
                "psychother": {'type': 'factor'},
                "fibrosisandother": {'type': 'factor'},
                "malnutrition": {'type': 'factor'},
                "hemo": {'type': 'factor'},
                "hematocrit": {'type': 'numeric'},
                "neutrophils": {'type': 'numeric'},
                "sodium": {'type': 'numeric'},
                "glucose": {'type': 'numeric'},
                "bloodureanitro": {'type': 'numeric'},
                "creatinine": {'type': 'numeric'},
                "bmi": {'type': 'numeric'},
                "pulse": {'type': 'numeric'},
                "respiration": {'type': 'numeric'},
                "secondarydiagnosisnonicd9": {'type': 'factor'},
                "discharged": {'type': 'character'},
                "facid": {'type': 'factor'},
                "lengthofstay": {'type': 'integer'}}
    return col_type_info



def create_database(connection_string, database_name):
    pyodbc_cnxn = pyodbc.connect(connection_string)
    pyodbc_cursor = pyodbc_cnxn.cursor()
    parameters = "if not exists(SELECT * FROM sys.databases WHERE name = '{}') CREATE DATABASE {};".format(database_name, database_name)
    pyodbc_cursor.execute(parameters)
    pyodbc_cursor.close()
    pyodbc_cnxn.commit()
    pyodbc_cnxn.close()


def display_head(table_name, n_rows):
    table_sql = RxSqlServerData(sql_query = "SELECT TOP({}}) * FROM {}}".format(n_rows, table_name), connection_string = connection_string)
    table = rx_import(table_sql)
    print(table)


def detect_table(table_name, connection_string):
    detect_sql = RxSqlServerData(sql_query="IF EXISTS (select 1 from information_schema.tables where table_name = '{}') SELECT 1 ELSE SELECT 0".format(table_name),
                                 connection_string=connection_string)
    does_exist = rx_import(detect_sql)
    if does_exist.iloc[0,0] == 1: return True
    else: return False


def drop_view(view_name, connection_string):
    pyodbc_cnxn = pyodbc.connect(connection_string)
    pyodbc_cursor = pyodbc_cnxn.cursor()
    pyodbc_cursor.execute("IF OBJECT_ID ('{}', 'V') IS NOT NULL DROP VIEW {} ;".format(view_name, view_name))
    pyodbc_cursor.close()
    pyodbc_cnxn.commit()
    pyodbc_cnxn.close()


def alter_column(table, column, data_type, connection_string):
    pyodbc_cnxn = pyodbc.connect(connection_string)
    pyodbc_cursor = pyodbc_cnxn.cursor()
    pyodbc_cursor.execute("ALTER TABLE {} ALTER COLUMN {} {};".format(table, column, data_type))
    pyodbc_cursor.close()
    pyodbc_cnxn.commit()
    pyodbc_cnxn.close()


def get_num_rows(table, connection_string):
    count_sql = RxSqlServerData(sql_query="SELECT COUNT(*) FROM {};".format(table), connection_string=connection_string)
    count = rx_import(count_sql)
    count = count.iloc[0,0]
    return count


def create_formula(response, features, to_remove=None):
    if to_remove is None:
        feats = [x for x in features if x not in [response]]
    else:
        feats = [x for x in features if x not in to_remove and x not in [response]]
    formula = "{} ~ ".format(response) + " + ".join(feats)
    return formula


def train_test_split(id, table, train_table, p, connection_string):
    pyodbc_cnxn = pyodbc.connect(connection_string)
    pyodbc_cursor = pyodbc_cnxn.cursor()
    pyodbc_cursor.execute("DROP TABLE if exists {};".format(train_table))
    pyodbc_cursor.execute("SELECT {} INTO {} FROM {} WHERE ABS(CAST(CAST(HashBytes('MD5', CAST({} AS varchar(10))) AS VARBINARY(64)) AS BIGINT) % 100) < {};".format(id, train_table, table, id, p))
    pyodbc_cursor.close()
    pyodbc_cnxn.commit()
    pyodbc_cnxn.close()


def insert_model(classifier, name, connection_string):
    classifier_odbc = RxOdbcData(connection_string, table="Models")
    rx_write_object(classifier_odbc, key=name, value=classifier, serialize=True, overwrite=True)


def retrieve_model(connection_string, name):
    classifier_odbc = RxOdbcData(connection_string, table="Models")
    classifier = rx_read_object(classifier_odbc, key=name, deserialize=True)
    return classifier


def evaluate_model(observed, predicted, model):
    mean_observed = np.mean(observed)
    se = (observed - predicted)**2
    ae = abs(observed - predicted)
    sem = (observed - mean_observed)**2
    aem = abs(observed - mean_observed)
    mae = np.mean(ae)
    rmse = np.sqrt(np.mean(se))
    rae = sum(ae) / sum(aem)
    rse = sum(se) / sum(sem)
    rsq = 1 - rse
    metrics = OrderedDict([ ("model_name", [model]),
				("mean_absolute_error", [mae]),
                ("root_mean_squared_error", [rmse]),
                ("relative_absolute_error", [rae]),
                ("relative_squared_error", [rse]),
                ("coefficient_of_determination", [rsq]) ])
    print(metrics)
    return metrics


def create_summary_dataframe(models_metrics):
    """ Create a summary panda DataFrame that combines all the metrics. """
    # Assuming all the metrics dictionary have the smae key.
    error_types = models_metrics[0].keys()
    summary = {}
    for model_metrics in models_metrics:
        for key in error_types:
            if key in summary:
                summary[key].append(model_metrics[key][0])
            else:
                summary[key] = model_metrics[key]
    return DataFrame(summary)

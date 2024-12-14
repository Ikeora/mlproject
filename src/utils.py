import os
import sys
from src.exception import CustomException
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import dill

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(x_train,y_train,x_test,y_test,models,params):
    try:
        report={}

        for model_name,model in models.items():

            param_grid=params.get(model_name,{})
            
            #fitting the model 
            if param_grid:
                model= GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    scoring='r2',
                    cv=5
                )


            model.fit(x_train,y_train)
            #predicting using train and test data
            y_train_pred= model.predict(x_train)

            y_test_pred= model.predict(x_test)

            # calculate r2 scores
            train_model_score= r2_score(y_train,y_train_pred)

            test_model_score= r2_score(y_test,y_test_pred)

            report[model_name]= test_model_score

        return report
        
    except Exception as e:
        raise CustomException(e,sys)   

def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)

        

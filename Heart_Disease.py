!pip install pgmpy
import numpy as np
import csv import pandas as pd 
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator 
from pgmpy.inference import VariableElimination

heartDisease=pd.read_csv('/heart_disease.csv') heartDisease=heartDisease.replace('?',np.nan)

print('Few examples from the dataset are given below') print(heartDisease.head())

model=BayesianModel([('age','trestbps'),('age','fbs'),('sex','trestbps'),('exang','trestbps'),('trestbps','heartDisease'),('fbs','heartDisease'),('heartDisease','restecg'),('heartDisease','thalach'),('heartDisease','chol')]) 
print("\n Learning CPD using MaximumLikelihoodEstimators") model.fit(heartDisease,estimator=MaximumLikelihoodEstimator) 
print("\n Inferencing with Bayesian Network") 
heartDisease_infer=VariableElimination(model) 
print("\n 1.Probability of HeartDisease given age=30") 
q=heartDisease_infer.query(variables=['heartDisease'],evidence={'age':63})
print(q) print("\n 2.Probability of HeartDisease given cholestrol=100") 
q=heartDisease_infer.query(variables=['heartDisease'],evidence={'chol':233}) 
print(q)

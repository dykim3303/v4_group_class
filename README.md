# DATA1030 (Fall 2024) -- Predicting Quarter Profitability in the Visegrad Group: A Study of Quarterly Data in 2017

## Introduction
### Problem Statement and Importance:
The Visegrad (V4) Group is comprised of 450 companies from Czechia, Hungary, Poland, and Slovakia. The dataset
from Tomczak's paper, "Ratio Selection between Six Sectors in the Visegrad Group Using Parametric and Nonparametric ANOVA"
describes the financial status of V4 companies via 82 financial ratios. Given the expansive range
of financial ratios/information, it would be advantageous to predict KPIs like profitability ratio
to inform budget allocatio/planning and ascertain comapny financial health. In this project, I 
develop a machine learning pipeline that predicts the next quarter's profitability ratio (X39: EBITDA*/sales
revenue) based on this year's quarterly financial information. 


### Data Source: 
UC Irvine Machine Learning Repository (https://archive.ics.uci.edu/dataset/830/visegrad+group+companies+data-2).

### Data Collection: 
The dataset was collected for the paper, "Ratio Selection between Six Sectors in the Visegrad Group Using Parametric and Nonparametric ANOVA" by 
Sebastian Tomczak, et al. 

### File Structure
data/: We store all raw and preprocessed data files here. The quarterly CSV files from 2017 and Q1 of 2018 were used in the source files, and the original .arff files were converted via the 
arffToCsv.py script. 
results/: Trained models and results are stored as pickle files. The results/models are stored in 2/3 dimensional arrays, indexed by: Lasso (0), Ridge (1), RandomForestRegressor (2).
report/: Final report detailing methodology, results discussion, and outlook are contained here.
src/: All the notebooks and codes containing preprocessing, EDA, machine learning pipeline are stored.

### Dependencies
python version: 3.10.12
package versions:
requirements = {'numpy': "1.24.3", 'matplotlib': "3.7.3",'sklearn': "1.2.2", 'mlxtend': "0.23.0",
'pandas': "2.0.3",'xgboost': "2.0.1", 'shap': "0.43.0", 'seaborn': "0.12.2",}

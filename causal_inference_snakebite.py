# importing required libraries
import os, warnings, random
import dowhy
import econml
from dowhy import CausalModel
import pandas as pd
import numpy as np
import econml
from econml.dml import DML
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from zepid.graphics import EffectMeasurePlot
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
import numpy as np, scipy.stats as st
import scipy.stats as stats
from zepid.causal.causalgraph import DirectedAcyclicGraph
from zepid.graphics import EffectMeasurePlot
import numpy as np, scipy.stats as st
from sklearn.linear_model import LassoCV
from econml.dml import CausalForestDML
from itertools import product
from econml.dml import SparseLinearDML
from econml.dml import LinearDML
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from econml.score import RScorer
from sklearn.model_selection import train_test_split
from econml.dml import CausalForestDML
from econml.dml import KernelDML
from joblib import Parallel, delayed
import warnings
from plotnine import ggplot, aes, geom_line, geom_ribbon, ggtitle, labs, geom_point, geom_hline
from econml.orf import DROrthoForest, DMLOrthoForest
from econml.utilities import WeightedModelWrapper
from xgboost import XGBRegressor, XGBClassifier


np.int = np.int32
np.float = np.float64
np.bool = np.bool_

# Set seeds to make the results more reproducible
def seed_everything(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

seed = 123
seed_everything(seed)
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: '%.2f' % x)


#%%#
#import data top300
data = pd.read_csv("D:/clases/UDES/articulo accidente ofidico/uniAndes/ci/manuscript/data_top300.csv", encoding='latin-1') 

data_col = data[['Code.DANE',
                 'excess', 'Rain',
                 'SST3', 'SST4', 'SST34', 'SST12', 
                 'Esoi', 'soi', 
                 'NATL', 'SATL', 'TROP', 
                 'forest', 'Rgdp',
                 ]]



#z-score
data_col.SST3 = stats.zscore(data_col.SST3, nan_policy='omit') 
data_col.SST4 = stats.zscore(data_col.SST4, nan_policy='omit')
data_col.SST34 = stats.zscore(data_col.SST34, nan_policy='omit') 
data_col.SST12 = stats.zscore(data_col.SST12, nan_policy='omit') 
data_col.Esoi = stats.zscore(data_col.Esoi, nan_policy='omit')
data_col.soi = stats.zscore(data_col.soi, nan_policy='omit') 
data_col.NATL = stats.zscore(data_col.NATL, nan_policy='omit')
data_col.SATL = stats.zscore(data_col.SATL, nan_policy='omit')  
data_col.TROP = stats.zscore(data.TROP, nan_policy='omit')
data_col.Rain = stats.zscore(data.Rain, nan_policy='omit')

data_col = data_col.dropna()

#####Rainfall
Y = data_col.excess.to_numpy() 
T = data_col.Rain.to_numpy()
W = data_col[['soi', 'Esoi', 'SST3', 'SST4', 'SST34', 'SST12', 'NATL', 'SATL', 'TROP', 'forest']].to_numpy().reshape(-1, 10)
X = data_col[['forest']].to_numpy().reshape(-1, 1)



## Ignore warnings
warnings.filterwarnings('ignore') 
reg1 = lambda: GradientBoostingClassifier(n_estimators=1500, random_state=123)
reg2 = lambda: GradientBoostingRegressor(n_estimators=1500, random_state=123)



#Model Selection for Causal Effect Model with the RScorer
# A multitude of possible approaches for CATE estimation under conditional exogeneity
X_train, X_val, T_train, T_val, Y_train, Y_val, W_train, W_val = train_test_split(X, T, Y, W, test_size=.4)



models = [
        ('ldml', LinearDML(model_y=reg1(), model_t=reg2(), discrete_treatment=False,
                             linear_first_stages=False, cv=3, random_state=123)),
        ('sldml', SparseLinearDML(model_y=reg1(), model_t=reg2(), discrete_treatment=False,
                                    featurizer=PolynomialFeatures(degree=3, include_bias=False),
                                    linear_first_stages=False, cv=3, random_state=123)),
        ('dml', DML(model_y=reg1(), model_t=reg2(), model_final=LassoCV(), discrete_treatment=False,
                             featurizer=PolynomialFeatures(degree=3),
                             linear_first_stages=False, cv=3, random_state=123)),
        ('forest', CausalForestDML(model_y=reg1(), model_t=reg2(), featurizer=PolynomialFeatures(degree=3), 
                                   discrete_treatment=False, cv=3, random_state=123)),       
        ('kernel', KernelDML(model_y=reg1(), model_t=reg2(),
                             cv=3, random_state=123))
               
         ]

def fit_model(name, model):
    return name, model.fit(Y_train, T_train, X=X_train, W=W_train)

models = Parallel(n_jobs=-1, verbose=1, backend="threading")(delayed(fit_model)(name, mdl) for name, mdl in models)

#Choose model with highest RScore
scorer = RScorer(model_y=reg1(), model_t=reg2(),
                 discrete_treatment=False, cv=3,
                 mc_iters=3, mc_agg='median')

scorer.fit(Y_val, T_val, X=X_val, W=W_val)


rscore = [scorer.score(mdl) for _, mdl in models]
print(rscore)#best model LinearDML

#%%#
#Estimation of ATE
estimate_Rainfall =  LinearDML(model_y=reg1(), model_t=reg2(), discrete_treatment=False,
                                    linear_first_stages=False, cv=3, random_state=123)

estimate_Rainfall = estimate_Rainfall.dowhy

# fit the model
estimate_Rainfall.fit(Y=Y, T=T, X=X, W=W, inference='statsmodels')

# predict effect for each sample X
estimate_Rainfall.effect(X) 

# ate
ate_Rainfall = estimate_Rainfall.ate(X) 
print(ate_Rainfall)

# confidence interval of ate
ci_Rainfall = estimate_Rainfall.ate_interval(X) 
print(ci_Rainfall)


#%%#
# constant marginal effect
#range of forest
min_forest = 0
max_forest = 100
delta = (max_forest - min_forest) / 100
X_test = np.arange(min_forest, max_forest + delta - 0.001, delta).reshape(-1, 1)

# Calculate marginal treatment effects
treatment_effects = estimate_Rainfall.const_marginal_effect(X_test)

# Calculate default (95%) marginal confidence intervals for the test data
te_upper, te_lower = estimate_Rainfall.const_marginal_effect_interval(X_test)

est2_Rainfall =  LinearDML(model_y=reg1(), model_t=reg2(), discrete_treatment=False,
                                    linear_first_stages=False, cv=3, random_state=123)

est2_Rainfall.fit(Y=Y, T=T, X=X, W=W, inference="statsmodels")

treatment_effects2 = est2_Rainfall.effect(X_test)
te_lower2_cons, te_upper2_cons = est2_Rainfall.effect_interval(X_test)


#Figure 4
(
ggplot(aes(x=X_test.flatten(), y=treatment_effects2)) 
  #+ geom_point() 
  + geom_line()
  + geom_ribbon(aes(ymin = te_lower2_cons, ymax = te_upper2_cons), alpha = .1)
  + labs(x='% of forest', y='Effect of rainfall on excess cases of snakebite')
  + geom_hline(aes(yintercept=0), color="red", linetype="dashed")
  
)  




#%%#
#Refute tests
#with random common cause
random_Rainfall = estimate_Rainfall.refute_estimate(method_name="random_common_cause", random_state=123)
print(random_Rainfall)

#with replace a random subset of the data
subset_Rainfall = estimate_Rainfall.refute_estimate(method_name="data_subset_refuter", subset_fraction=0.9, num_simulations=3)
print(subset_Rainfall)

#with placebo 
placebo_Rainfall = estimate_Rainfall.refute_estimate(method_name="placebo_treatment_refuter", placebo_type="permute", num_simulations=3)
print(placebo_Rainfall)



#%%#
#New estimation CATE for soi
Y = data_col.excess.to_numpy() 
T = data_col.Rain.to_numpy()
W = data_col[['soi', 'Esoi', 'SST3', 'SST4', 'SST34', 'SST12', 'NATL', 'SATL', 'TROP', 'forest']].to_numpy().reshape(-1, 10)
X = data_col[['soi']].to_numpy().reshape(-1, 1)

#Step 3: Estimation of the effect 
estimate_Rainfall =  LinearDML(model_y=reg1(), model_t=reg2(), discrete_treatment=False,
                                    linear_first_stages=False, cv=3, random_state=123)

estimate_Rainfall = estimate_Rainfall.dowhy

# fit the CATE model
estimate_Rainfall.fit(Y=Y, T=T, X=X, W=W, inference='statsmodels')  

#CATE
#range of soi
min_Rgdp = -2.6
max_Rgdp = 2.7
delta = (max_Rgdp - min_Rgdp) / 100
X_test = np.arange(min_Rgdp, max_Rgdp + delta - 0.001, delta).reshape(-1, 1)

# Calculate marginal treatment effects
treatment_effects = estimate_Rainfall.const_marginal_effect(X_test)

# Calculate default (95%) marginal confidence intervals for the test data
te_upper, te_lower = estimate_Rainfall.const_marginal_effect_interval(X_test)

est2_Rainfall =  LinearDML(model_y=reg1(), model_t=reg2(), discrete_treatment=False,
                                    linear_first_stages=False, cv=3, random_state=123)

est2_Rainfall.fit(Y=Y, T=T, X=X, inference="statsmodels")

treatment_effects2 = est2_Rainfall.effect(X_test)
te_lower2_cons, te_upper2_cons = est2_Rainfall.effect_interval(X_test)


#Figure Supplement 1A
(
ggplot(aes(x=X_test.flatten(), y=treatment_effects2)) 
  + geom_line() 
  + geom_ribbon(aes(ymin = te_lower2_cons, ymax = te_upper2_cons), alpha = .1)
  + labs(x='soi (sd)', y='Effect of rainfall on excess of cases of snakebite')
  + geom_hline(yintercept = 0)
)  



#%%#
#New estimation CATE for Esoi
Y = data_col.excess.to_numpy() 
T = data_col.Rain.to_numpy()
W = data_col[['soi', 'Esoi', 'SST3', 'SST4', 'SST34', 'SST12', 'NATL', 'SATL', 'TROP', 'forest']].to_numpy().reshape(-1, 10)
X = data_col[['Esoi']].to_numpy().reshape(-1, 1)

#Step 3: Estimation of the effect 
estimate_Rainfall =  LinearDML(model_y=reg1(), model_t=reg2(), discrete_treatment=False,
                                    linear_first_stages=False, cv=3, random_state=123)

estimate_Rainfall = estimate_Rainfall.dowhy

# fit the CATE model
estimate_Rainfall.fit(Y=Y, T=T, X=X, W=W, inference='statsmodels')  

#CATE
#range of Esoi
min_Rgdp = -3.4
max_Rgdp = 2.7
delta = (max_Rgdp - min_Rgdp) / 100
X_test = np.arange(min_Rgdp, max_Rgdp + delta - 0.001, delta).reshape(-1, 1)

# Calculate marginal treatment effects
treatment_effects = estimate_Rainfall.const_marginal_effect(X_test)

# Calculate default (95%) marginal confidence intervals for the test data
te_upper, te_lower = estimate_Rainfall.const_marginal_effect_interval(X_test)

est2_Rainfall =  LinearDML(model_y=reg1(), model_t=reg2(), discrete_treatment=False,
                                    linear_first_stages=False, cv=3, random_state=123)

est2_Rainfall.fit(Y=Y, T=T, X=X, inference="statsmodels")

treatment_effects2 = est2_Rainfall.effect(X_test)
te_lower2_cons, te_upper2_cons = est2_Rainfall.effect_interval(X_test)


#Figure Supplement 1B
(
ggplot(aes(x=X_test.flatten(), y=treatment_effects2)) 
  + geom_line() 
  + geom_ribbon(aes(ymin = te_lower2_cons, ymax = te_upper2_cons), alpha = .1)
  + labs(x='Esoi (sd)', y='Effect of rainfall on excess of cases of snakebite')
  + geom_hline(yintercept = 0)
  
) 


#%%#
#New estimation CATE for SST3
Y = data_col.excess.to_numpy() 
T = data_col.Rain.to_numpy()
W = data_col[['soi', 'Esoi', 'SST3', 'SST4', 'SST34', 'SST12', 'NATL', 'SATL', 'TROP', 'forest']].to_numpy().reshape(-1, 10)
X = data_col[['SST3']].to_numpy().reshape(-1, 1)

#Step 3: Estimation of the effect 
estimate_Rainfall =  LinearDML(model_y=reg1(), model_t=reg2(), discrete_treatment=False,
                                    linear_first_stages=False, cv=3, random_state=123)

estimate_Rainfall = estimate_Rainfall.dowhy

# fit the CATE model
estimate_Rainfall.fit(Y=Y, T=T, X=X, W=W, inference='statsmodels')  

#CATE
#range of SST3
min_Rgdp = -1.9
max_Rgdp = 2.1
delta = (max_Rgdp - min_Rgdp) / 100
X_test = np.arange(min_Rgdp, max_Rgdp + delta - 0.001, delta).reshape(-1, 1)

# Calculate marginal treatment effects
treatment_effects = estimate_Rainfall.const_marginal_effect(X_test)

# Calculate default (95%) marginal confidence intervals for the test data
te_upper, te_lower = estimate_Rainfall.const_marginal_effect_interval(X_test)

est2_Rainfall =  LinearDML(model_y=reg1(), model_t=reg2(), discrete_treatment=False,
                                    linear_first_stages=False, cv=3, random_state=123)

est2_Rainfall.fit(Y=Y, T=T, X=X, inference="statsmodels")

treatment_effects2 = est2_Rainfall.effect(X_test)
te_lower2_cons, te_upper2_cons = est2_Rainfall.effect_interval(X_test)


#Figure Supplement 1C
(
ggplot(aes(x=X_test.flatten(), y=treatment_effects2)) 
  + geom_line() 
  + geom_ribbon(aes(ymin = te_lower2_cons, ymax = te_upper2_cons), alpha = .1)
  + labs(x='SST3 (sd)', y='Effect of rainfall on excess of cases of snakebite')
  + geom_hline(yintercept = 0)
  
) 


#%%#
#New estimation CATE for SST4
Y = data_col.excess.to_numpy() 
T = data_col.Rain.to_numpy()
W = data_col[['soi', 'Esoi', 'SST3', 'SST4', 'SST34', 'SST12', 'NATL', 'SATL', 'TROP', 'forest']].to_numpy().reshape(-1, 10)
X = data_col[['SST4']].to_numpy().reshape(-1, 1)

#Step 3: Estimation of the effect 
estimate_Rainfall =  LinearDML(model_y=reg1(), model_t=reg2(), discrete_treatment=False,
                                    linear_first_stages=False, cv=3, random_state=123)

estimate_Rainfall = estimate_Rainfall.dowhy

# fit the CATE model
estimate_Rainfall.fit(Y=Y, T=T, X=X, W=W, inference='statsmodels')  

#CATE
#range of SST4
min_Rgdp = -2.6
max_Rgdp = 2.3
delta = (max_Rgdp - min_Rgdp) / 100
X_test = np.arange(min_Rgdp, max_Rgdp + delta - 0.001, delta).reshape(-1, 1)

# Calculate marginal treatment effects
treatment_effects = estimate_Rainfall.const_marginal_effect(X_test)

# Calculate default (95%) marginal confidence intervals for the test data
te_upper, te_lower = estimate_Rainfall.const_marginal_effect_interval(X_test)

est2_Rainfall =  LinearDML(model_y=reg1(), model_t=reg2(), discrete_treatment=False,
                                    linear_first_stages=False, cv=3, random_state=123)

est2_Rainfall.fit(Y=Y, T=T, X=X, inference="statsmodels")

treatment_effects2 = est2_Rainfall.effect(X_test)
te_lower2_cons, te_upper2_cons = est2_Rainfall.effect_interval(X_test)


#Figure Supplement 1D
(
ggplot(aes(x=X_test.flatten(), y=treatment_effects2)) 
  + geom_line() 
  + geom_ribbon(aes(ymin = te_lower2_cons, ymax = te_upper2_cons), alpha = .1)
  + labs(x='SST4 (sd)', y='Effect of rainfall on excess of cases of snakebite')
  + geom_hline(yintercept = 0)
  
) 


#%%#
#New estimation CATE for SST34
Y = data_col.excess.to_numpy() 
T = data_col.Rain.to_numpy()
W = data_col[['soi', 'Esoi', 'SST3', 'SST4', 'SST34', 'SST12', 'NATL', 'SATL', 'TROP', 'forest']].to_numpy().reshape(-1, 10)
X = data_col[['SST34']].to_numpy().reshape(-1, 1)

#Step 3: Estimation of the effect 
estimate_Rainfall =  LinearDML(model_y=reg1(), model_t=reg2(), discrete_treatment=False,
                                    linear_first_stages=False, cv=3, random_state=123)

estimate_Rainfall = estimate_Rainfall.dowhy

# fit the CATE model
estimate_Rainfall.fit(Y=Y, T=T, X=X, W=W, inference='statsmodels')  

#CATE
#range of SST34
min_Rgdp = -2.1
max_Rgdp = 2.4
delta = (max_Rgdp - min_Rgdp) / 100
X_test = np.arange(min_Rgdp, max_Rgdp + delta - 0.001, delta).reshape(-1, 1)

# Calculate marginal treatment effects
treatment_effects = estimate_Rainfall.const_marginal_effect(X_test)

# Calculate default (95%) marginal confidence intervals for the test data
te_upper, te_lower = estimate_Rainfall.const_marginal_effect_interval(X_test)

est2_Rainfall =  LinearDML(model_y=reg1(), model_t=reg2(), discrete_treatment=False,
                                    linear_first_stages=False, cv=3, random_state=123)

est2_Rainfall.fit(Y=Y, T=T, X=X, inference="statsmodels")

treatment_effects2 = est2_Rainfall.effect(X_test)
te_lower2_cons, te_upper2_cons = est2_Rainfall.effect_interval(X_test)


#Figure Supplement 1E
(
ggplot(aes(x=X_test.flatten(), y=treatment_effects2)) 
  + geom_line() 
  + geom_ribbon(aes(ymin = te_lower2_cons, ymax = te_upper2_cons), alpha = .1)
  + labs(x='SST34 (sd)', y='Effect of rainfall on excess of cases of snakebite')
  + geom_hline(yintercept = 0)
  
)


#%%#
#New estimation CATE for SST12
Y = data_col.excess.to_numpy() 
T = data_col.Rain.to_numpy()
W = data_col[['soi', 'Esoi', 'SST3', 'SST4', 'SST34', 'SST12', 'NATL', 'SATL', 'TROP', 'forest']].to_numpy().reshape(-1, 10)
X = data_col[['SST12']].to_numpy().reshape(-1, 1)

#Step 3: Estimation of the effect 
estimate_Rainfall =  LinearDML(model_y=reg1(), model_t=reg2(), discrete_treatment=False,
                                    linear_first_stages=False, cv=3, random_state=123)

estimate_Rainfall = estimate_Rainfall.dowhy

# fit the CATE model
estimate_Rainfall.fit(Y=Y, T=T, X=X, W=W, inference='statsmodels')  

#CATE
#range of SST12
min_Rgdp = -1.9
max_Rgdp = 2.2
delta = (max_Rgdp - min_Rgdp) / 100
X_test = np.arange(min_Rgdp, max_Rgdp + delta - 0.001, delta).reshape(-1, 1)

# Calculate marginal treatment effects
treatment_effects = estimate_Rainfall.const_marginal_effect(X_test)

# Calculate default (95%) marginal confidence intervals for the test data
te_upper, te_lower = estimate_Rainfall.const_marginal_effect_interval(X_test)

est2_Rainfall =  LinearDML(model_y=reg1(), model_t=reg2(), discrete_treatment=False,
                                    linear_first_stages=False, cv=3, random_state=123)

est2_Rainfall.fit(Y=Y, T=T, X=X, inference="statsmodels")

treatment_effects2 = est2_Rainfall.effect(X_test)
te_lower2_cons, te_upper2_cons = est2_Rainfall.effect_interval(X_test)


#Figure Supplement 1F
(
ggplot(aes(x=X_test.flatten(), y=treatment_effects2)) 
  + geom_line() 
  + geom_ribbon(aes(ymin = te_lower2_cons, ymax = te_upper2_cons), alpha = .1)
  + labs(x='SST12 (sd)', y='Effect of rainfall on excess of cases of snakebite')
  + geom_hline(yintercept = 0)
  
)



#%%#
#New estimation CATE for NATL
Y = data_col.excess.to_numpy() 
T = data_col.Rain.to_numpy()
W = data_col[['soi', 'Esoi', 'SST3', 'SST4', 'SST34', 'SST12', 'NATL', 'SATL', 'TROP', 'forest']].to_numpy().reshape(-1, 10)
X = data_col[['NATL']].to_numpy().reshape(-1, 1)

#Step 3: Estimation of the effect 
estimate_Rainfall =  LinearDML(model_y=reg1(), model_t=reg2(), discrete_treatment=False,
                                    linear_first_stages=False, cv=3, random_state=123)

estimate_Rainfall = estimate_Rainfall.dowhy

# fit the CATE model
estimate_Rainfall.fit(Y=Y, T=T, X=X, W=W, inference='statsmodels')  

#CATE
#range of NATL
min_Rgdp = -1.7
max_Rgdp = 1.7
delta = (max_Rgdp - min_Rgdp) / 100
X_test = np.arange(min_Rgdp, max_Rgdp + delta - 0.001, delta).reshape(-1, 1)

# Calculate marginal treatment effects
treatment_effects = estimate_Rainfall.const_marginal_effect(X_test)

# Calculate default (95%) marginal confidence intervals for the test data
te_upper, te_lower = estimate_Rainfall.const_marginal_effect_interval(X_test)

est2_Rainfall =  LinearDML(model_y=reg1(), model_t=reg2(), discrete_treatment=False,
                                    linear_first_stages=False, cv=3, random_state=123)

est2_Rainfall.fit(Y=Y, T=T, X=X, inference="statsmodels")

treatment_effects2 = est2_Rainfall.effect(X_test)
te_lower2_cons, te_upper2_cons = est2_Rainfall.effect_interval(X_test)


#Figure Supplement 1G
(
ggplot(aes(x=X_test.flatten(), y=treatment_effects2)) 
  + geom_line() 
  + geom_ribbon(aes(ymin = te_lower2_cons, ymax = te_upper2_cons), alpha = .1)
  + labs(x='NATL (sd)', y='Effect of rainfall on excess of cases of snakebite')
  + geom_hline(yintercept = 0)
  
)



#%%#
#New estimation CATE for SATL
Y = data_col.excess.to_numpy() 
T = data_col.Rain.to_numpy()
W = data_col[['soi', 'Esoi', 'SST3', 'SST4', 'SST34', 'SST12', 'NATL', 'SATL', 'TROP', 'forest']].to_numpy().reshape(-1, 10)
X = data_col[['SATL']].to_numpy().reshape(-1, 1)

#Step 3: Estimation of the effect 
estimate_Rainfall =  LinearDML(model_y=reg1(), model_t=reg2(), discrete_treatment=False,
                                    linear_first_stages=False, cv=3, random_state=123)

estimate_Rainfall = estimate_Rainfall.dowhy

# fit the CATE model
estimate_Rainfall.fit(Y=Y, T=T, X=X, W=W, inference='statsmodels')  

#CATE
#range of SATL
min_Rgdp = -1.5
max_Rgdp = 1.8
delta = (max_Rgdp - min_Rgdp) / 100
X_test = np.arange(min_Rgdp, max_Rgdp + delta - 0.001, delta).reshape(-1, 1)

# Calculate marginal treatment effects
treatment_effects = estimate_Rainfall.const_marginal_effect(X_test)

# Calculate default (95%) marginal confidence intervals for the test data
te_upper, te_lower = estimate_Rainfall.const_marginal_effect_interval(X_test)

est2_Rainfall =  LinearDML(model_y=reg1(), model_t=reg2(), discrete_treatment=False,
                                    linear_first_stages=False, cv=3, random_state=123)

est2_Rainfall.fit(Y=Y, T=T, X=X, inference="statsmodels")

treatment_effects2 = est2_Rainfall.effect(X_test)
te_lower2_cons, te_upper2_cons = est2_Rainfall.effect_interval(X_test)


#Figure Supplement 1H
(
ggplot(aes(x=X_test.flatten(), y=treatment_effects2)) 
  + geom_line() 
  + geom_ribbon(aes(ymin = te_lower2_cons, ymax = te_upper2_cons), alpha = .1)
  + labs(x='SATL (sd)', y='Effect of rainfall on excess of cases of snakebite')
  + geom_hline(yintercept = 0)
)



#%%#
#New estimation CATE for TROP
Y = data_col.excess.to_numpy() 
T = data_col.Rain.to_numpy()
W = data_col[['soi', 'Esoi', 'SST3', 'SST4', 'SST34', 'SST12', 'NATL', 'SATL', 'TROP', 'forest']].to_numpy().reshape(-1, 10)
X = data_col[['TROP']].to_numpy().reshape(-1, 1)

#Step 3: Estimation of the effect 
estimate_Rainfall =  LinearDML(model_y=reg1(), model_t=reg2(), discrete_treatment=False,
                                    linear_first_stages=False, cv=3, random_state=123)

estimate_Rainfall = estimate_Rainfall.dowhy

# fit the CATE model
estimate_Rainfall.fit(Y=Y, T=T, X=X, W=W, inference='statsmodels')  

#CATE
#range of TROP
min_Rgdp = -1.7
max_Rgdp = 2.5
delta = (max_Rgdp - min_Rgdp) / 100
X_test = np.arange(min_Rgdp, max_Rgdp + delta - 0.001, delta).reshape(-1, 1)

# Calculate marginal treatment effects
treatment_effects = estimate_Rainfall.const_marginal_effect(X_test)

# Calculate default (95%) marginal confidence intervals for the test data
te_upper, te_lower = estimate_Rainfall.const_marginal_effect_interval(X_test)

est2_Rainfall =  LinearDML(model_y=reg1(), model_t=reg2(), discrete_treatment=False,
                                    linear_first_stages=False, cv=3, random_state=123)

est2_Rainfall.fit(Y=Y, T=T, X=X, inference="statsmodels")

treatment_effects2 = est2_Rainfall.effect(X_test)
te_lower2_cons, te_upper2_cons = est2_Rainfall.effect_interval(X_test)


#Figure Supplement 1I
(
ggplot(aes(x=X_test.flatten(), y=treatment_effects2)) 
  + geom_line() 
  + geom_ribbon(aes(ymin = te_lower2_cons, ymax = te_upper2_cons), alpha = .1)
  + labs(x='TROP (sd)', y='Effect of rainfall on excess of cases of snakebite')
  + geom_hline(yintercept = 0)
  
)





















































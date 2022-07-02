import matplotlib.pyplot       as     mpl
import numpy                   as     np
import pandas                  as     pd
from   sklearn.ensemble        import RandomForestClassifier
from   sklearn.metrics         import accuracy_score         as Acc
from   sklearn.model_selection import train_test_split       as ShuffleSplit
from   ICP.Models              import ICPRuleEnsemble
import time

# %% Load data
DF = pd.read_csv('TitanicFull.csv')
# %% Create features
DF['IsFemale']  = DF.Sex      == 'female'

DF['CabinC']    = DF.Cabin.str.contains('C').fillna(0)
DF['CabinE']    = DF.Cabin.str.contains('C').fillna(0)
DF['CabinD']    = DF.Cabin.str.contains('C').fillna(0)
DF['CabinB']    = DF.Cabin.str.contains('C').fillna(0)
DF['CabinF']    = DF.Cabin.str.contains('C').fillna(0)

ttl = DF.Name.map(lambda x : x.split(', ', 1)[-1].split(' ', 1)[0].strip())
DF['IsMr']       = ttl == 'Mr.'
DF['IsMiss']     = ttl == 'Miss.'
DF['IsMrs']      = ttl == 'Mrs.'
DF['IsMaster']   = ttl == 'Master'
DF['IsOthTitle'] = ~ttl.isin(('Mr.', 'Miss.', 'Mrs.', 'Master'))

DF['EmbarkedS']  = DF.Embarked == 'S'
DF['EmbarkedC']  = DF.Embarked == 'C'
DF['EmbarkedQ']  = DF.Embarked == 'Q'

DF['1stClass']   = DF.Pclass == 1
DF['2ndClass']   = DF.Pclass == 2
DF['3rdClass']   = DF.Pclass == 3
# %% Select columns to use
skipCol  = ('Survived', 'PassengerId')
FEAT_COL = DF.dtypes[(DF.dtypes != 'object') & (~DF.columns.isin(skipCol))].index
print('\n'.join(FEAT_COL))
# %% Setup data problem
np.random.seed(0)
trn, tst = ShuffleSplit(np.arange(DF.shape[0]))

fv = DF[FEAT_COL].iloc[trn].median()
A  = DF[FEAT_COL].fillna(fv).astype('double').values
Y  = (DF['Survived'] == 1).values
W  = np.ones(Y.shape[0])
# %% Train ICPRE model
t1  = time.time()
IRE = ICPRuleEnsemble(mrg=1.0, maxIter=1000, lr=1.2345, cOrd='r', tm='gbc', nsd=0, xsd=16,
                      tmPar=dict(max_depth=1, min_impurity_decrease=1e-8), v=2)
IRE.fit(A[trn], Y[trn], W[trn])
t2  = time.time()
YP  = IRE.predict_proba(A)
YH  = YP.argmax(1)
# %% Print explain output
pn   = 258
pExp = IRE.Explain(A[[pn]], FEAT_COL)[0]

for ri, v in sorted(pExp, key=lambda x : x[1]):
    print('{:+.2f} {:s}'.format(v, ri))
# %% Print original rules
print('\nRules')
rl, cv = zip(*IRE.GetRules(FEAT_COL, False))
for i, (rli, vi) in enumerate(zip(rl, cv)):
   print('Rule {:3d}:'.format(i))
   print(' -Coef: {:+.6f}'.format(float(vi)))
   print(' -Pred: {:s}'.format(rli))
# %% Print consolidated rules
print('\nConsolidated Rules')
rl, cv = zip(*IRE.GetRules(FEAT_COL))
for i, (rli, vi) in enumerate(sorted(zip(rl, cv), key=lambda x : x[1])):
   print('Rule {:3d}:'.format(i))
   print(' -Coef: {:+.6f}'.format(float(vi)))
   print(' -Pred: {:s}'.format(rli))
# %% Print test results
print('\nICPRE:')
print('Trn: {:7.2%}'.format(Acc(Y[trn], YH[trn], sample_weight=W[trn])))
print('Tst: {:7.2%}'.format(Acc(Y[tst], YH[tst], sample_weight=W[tst])))
print('Elp: {:7.2f}s'.format(t2 - t1))
# %% Show response curves
for f in set(IRE.FA):
   x, y, isOrig = map(list, zip(*IRE.GetResponseCurve(f)))
   xr = (x[-2] - x[1]) 
   x[ 0] = x[ 1] - xr * 0.05
   x[-1] = x[-2] + xr * 0.05
   fig, ax = mpl.subplots()
   ax.plot(x, y)
   ax.set_xlabel(FEAT_COL[f])
   ax.set_ylabel('Change in Response')
   ax.set_title('{:s} Response Curve'.format(FEAT_COL[f]))
   mpl.show()
# %% Compare to RFC
t1  = time.time()
rfc = RandomForestClassifier().fit(A[trn], Y[trn], W[trn])
t2  = time.time()
YP2 = rfc.predict_proba(A)
YH2 = YP2.argmax(1)
print('\nRFC:')
print('Trn: {:7.2%}'.format(Acc(Y[trn], YH2[trn], sample_weight=W[trn])))
print('Tst: {:7.2%}'.format(Acc(Y[tst], YH2[tst], sample_weight=W[tst])))
print('Elp: {:6.2f}s'.format(t2 - t1))
print('Nod: {:7d}'.format(sum(i.tree_.node_count for i in rfc.estimators_)))

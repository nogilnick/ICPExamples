import numpy as np
from   ICP.Binning import KBDisc, OHToDense
from   sklearn.preprocessing import KBinsDiscretizer
# %% Unit tests
if __name__ == '__main__':

   for i in range(100):
      A = np.random.rand(100, 4)

      kbd2 = KBinsDiscretizer(10, encode='ordinal')
      B2   = kbd2.fit_transform(A)

      kbd = KBDisc(10).fit(A)
      B1 = OHToDense(kbd.transform(A), kbd.nBins)

      if np.abs(B1 - B2).max() > 1.1:
         raise Exception('Blah')
   # %% Test add/remove const functionality
   kbd = KBDisc(10, encode='onehot-dense', const=True)
   OHc = kbd.fit_transform(A)
   kbd.RemoveConst()
   OHn = kbd.transform(A)

   for _ in range(10):
      kbd.AddConst()
      OHi = kbd.transform(A)
      assert((OHi == OHc).all())
      kbd.RemoveConst()
      OHi = kbd.transform(A)
      assert((OHi == OHn).all())
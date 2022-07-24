from    ICP.PathSearch  import PatherAbs, PatherHinge
from    ICP.Solver      import ErrAbs, EPS, DERR_MAX, DEL, ErrHinge
import  numpy           as     np
from    time            import time

def SlowCheck(A, Y, X, W, d, errFx):
   # Distance until target is hit
   BV   = d * (Y - X) / A
   # Negative means moving away from target in this direction
   BV   = BV[BV >= 0]
   berr = errFx(X, Y, W)
   bdst = 0.
   for bvi in BV:
      eri = errFx(X + d * bvi * A, Y, W)
      if eri < berr:
         berr = eri
         bdst = bvi
   return berr, bdst
# %% Hinge loss
##########################################################################################

# HINGE VALUE TESTS

##########################################################################################
# %% Consistent across runs
eps0 = EPS
eps1 = DERR_MAX
eps2 = DEL

tel  = 0.
# %% Make sure ICP can find optimal distances for easy problems
# Pre-allocate temporary vectors for search
n  = 30
PT = PatherHinge(n, eps0, eps1, eps2)

elp = 0.
for i in range(999):
   Y    = np.random.choice([-1., 1.], size=n)
   A    = Y  # Column has correct direction for each sample
   X    = 2 * np.random.rand(n) - 1
   W    = np.full(n, 1 / n)
   f    = 0
   d    = +1
   vMax = n + 1

   err  = ErrHinge(X, Y, W)

   t0 = time()
   PT.FindDist(A, n, Y, W, X, d, vMax)
   elp += time() - t0

   errRed, bDist, bSlack = PT.GetResults()

   # Actual error reduction
   nErr   = ErrHinge(X + bDist * A, Y, W)
   actRed = nErr - err

   dTru = np.abs(X-Y).max()
   assert(np.isclose(nErr, 0))
   assert(np.isclose(actRed, errRed))
   assert(dTru <= bDist)
   assert(np.isclose(vMax - bDist, bSlack))

tel += elp
print('Elapsed: {:.2f}'.format(elp))
# %% Check that values beyond vMax are handled correctly
# Pre-allocate temporary vectors for search
n  = 30
PT = PatherHinge(n, eps0, eps1, eps2)

elp = 0.
for i in range(999):
   Y    = np.random.choice([-1., 1.], size=n)
   A    = Y  # Column has correct direction for each sample
   Y   *= 10
   X    = 2 * np.random.rand(n) - 1
   W    = np.full(n, 1 / n)
   f    = 0
   d    = +1
   vMax = 0.5

   err  = ErrHinge(X, Y, W)

   t0 = time()
   PT.FindDist(A, n, Y, W, X, d, vMax)
   elp += time() - t0

   errRed, bDist, bSlack = PT.GetResults()

   # Actual error reduction
   actRed = ErrHinge(X + bDist * A, Y, W) - err

   assert(np.isclose(actRed, errRed))
   assert(np.isclose(vMax, bDist))
   assert(np.isclose(vMax - bDist, bSlack))

tel += elp
print('Elapsed: {:.2f}'.format(elp))
# %% Check that optimal value is found when all directions are wrong
# Pre-allocate temporary vectors for search
n  = 30
PT = PatherHinge(n, eps0, eps1, eps2)

elp = 0.
for i in range(999):
   Y    = np.random.choice([-1., 1.], size=n)
   A    = Y
   X    = 2 * np.random.rand(n) - 1
   W    = np.full(n, 1 / n)
   f    = 0
   d    = -1  # Column has wrong direction for each sample
   vMax = 0.5

   err  = ErrHinge(X, Y, W)

   t0 = time()
   PT.FindDist(A, n, Y, W, X, d, vMax)
   elp += time() - t0

   errRed, bDist, bSlack = PT.GetResults()

   # Actual error reduction
   actRed = ErrHinge(X + bDist * A, Y, W) - err

   assert(np.isclose(actRed, errRed))
   assert(np.isclose(0., bDist))
   assert(np.isclose(vMax - bDist, bSlack))
tel += elp
print('Elapsed: {:.2f}'.format(elp))
# %% Compare random instance to brute force solution
# Pre-allocate temporary vectors for search
n  = 30
PT = PatherHinge(n, eps0, eps1, eps2)

elp = 0.
for i in range(999):
   Y    = np.random.choice([-1., 1.], size=n)
   A    = np.random.choice([-1., 1.], size=n)
   X    = 2 * np.random.rand(n) - 1
   W    = np.random.rand(n)
   W   /= W.sum()
   W    = np.full(n, 1 / n)
   f    = 0
   d    = np.random.choice([-1, 1])
   vMax = 10

   err  = ErrHinge(X, Y, W)

   t0 = time()
   PT.FindDist(A, n, Y, W, X, d, vMax)
   elp += time() - t0

   errRed, bDist, bSlack = PT.GetResults()

   if errRed >= 0:   # No error reduction possible in this dir; mark 0
      bDist  = 0
      errRed = 0
      bSlack = vMax

   # Actual error reduction
   nErr   = ErrHinge(X + d * bDist * A, Y, W)
   actRed = nErr - err

   # Brute force for optimal solution
   bErr_slow, bDst_slow = SlowCheck(A, Y, X, W, d, ErrHinge)
   sErr = bErr_slow - err

   assert(np.isclose(actRed, errRed))
   assert(np.isclose(sErr, actRed))
   assert(np.isclose(vMax - bDist, bSlack))
tel += elp
print('Elapsed: {:.2f}'.format(elp))
# %% Hinge loss time trial
n  = 50000
PT = PatherHinge(n, eps0, eps1, eps2)

tml = []
elp = 0.
for i in range(999):
   Y    = np.random.choice([-1., 1.], size=n)
   A    = np.random.choice([-1., 1.], size=n)
   X    = 2 * np.random.rand(n) - 1
   W    = np.random.rand(n)
   W   /= W.sum()
   W    = np.full(n, 1 / n)
   f    = 0
   d    = np.random.choice([-1, 1])
   vMax = 20

   err  = ErrHinge(X, Y, W)

   t0 = time()
   PT.FindDistCg(A, Y, W, X, d, vMax)
   tml.append(time() - t0)

elp  = sum(tml)
tdv  = np.std(tml)
tel += elp
print('Elapsed: {:.4f}'.format(elp))
print('Mean:    {:.4f}'.format(elp / len(tml)))
print('Std:     {:.4f}'.format(tdv))
# %% Abs loss
##########################################################################################

# ABS VALUE TESTS

##########################################################################################
# %% Compare random instance to brute force solution
# Pre-allocate temporary vectors for search
n  = 30
PT = PatherAbs(n, eps0)

elp = 0.
for i in range(999):
   Y    = np.random.choice([-1., 1.], size=n)
   A    = np.random.choice([-1., 1.], size=n)
   X    = 2 * np.random.rand(n) - 1
   W    = np.random.rand(n)
   W   /= W.sum()
   W    = np.full(n, 1 / n)
   f    = 0
   d    = np.random.choice([-1, 1])
   vMax = 10

   err  = ErrHinge(X, Y, W)

   t0 = time()
   PT.FindDistCg(A, Y, W, X, d, vMax)
   elp += time() -t0

   errRed, bDist, bSlack = PT.GetResults()

   if errRed >= 0:   # No error reduction possible in this dir; mark 0
      bDist  = 0
      errRed = 0
      bSlack = vMax

   # Actual error reduction
   nErr   = ErrAbs(X + d * bDist * A, Y, W)
   actRed = nErr - err

   # Brute force for optimal solution
   bErr_slow, bDst_slow = SlowCheck(A, Y, X, W, d, ErrAbs)
   sErr = bErr_slow - err

   assert(np.isclose(actRed, errRed))
   assert(np.isclose(sErr, actRed))
   assert(np.isclose(vMax - bDist, bSlack))
tel += elp
print('Elapsed: {:.2f}'.format(elp))
# %% Absolute value time trial
n  = 60000
PT = PatherAbs(n, eps0)


tml = []
elp = 0.
for i in range(999):
   Y    = np.random.choice([-1., 1.], size=n)
   A    = np.random.choice([-1., 1.], size=n)
   X    = 2 * np.random.rand(n) - 1
   W    = np.random.rand(n)
   W   /= W.sum()
   W    = np.full(n, 1 / n)
   f    = 0
   d    = np.random.choice([-1, 1])
   vMax = 10

   err  = ErrHinge(X, Y, W)

   t0 = time()
   PT.FindDistCg(A, Y, W, X, d, vMax)
   tml.append(time() - t0)

   errRed, bDist, bSlack = PT.GetResults()

elp  = sum(tml)
tdv  = np.std(tml)
tel += elp
print('Elapsed: {:.4f}'.format(elp))
print('Mean:    {:.4f}'.format(elp / len(tml)))
print('Std:     {:.4f}'.format(tdv))

# %% Print result
print('Tot Elp: {:.2f}'.format(elp))
print('SUCCESS')
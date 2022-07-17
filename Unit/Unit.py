import  numpy           as     np
from    ICP.PathSearch  import FindDist
from    ICP.Solver      import EPS, DERR_MAX, DEL, ErrHinge

def SlowCheck(A, Y, X, W, d):
   # Distance until target is hit
   BV = d * (Y - X) / A
   # Negative means moving away from target in this direction
   BV = BV[BV >= 0]
   berr = ErrHinge(X, Y, W)
   bdst = 0.
   for bvi in BV:
      eri = ErrHinge(X + d * bvi * A, Y, W)
      if eri < berr:
         berr = eri
         bdst = bvi
   return berr, bdst
# %% Consistent across runs
eps0 = EPS
eps1 = DERR_MAX
eps2 = DEL
# %% Make sure ICP can find optimal distances for easy problems
# Pre-allocate temporary vectors for search
n    = 10
BVae = np.empty(n + 1)
AWae = np.empty(n)
AEsi = np.empty(n, np.intp)

BVse = np.empty(n + 1)
AWse = np.empty(n)
SEsi = np.empty(n, np.intp)

tmp = np.empty(n)

rvs = np.empty(3) # Return values
   
for i in range(999):
   Y    = np.random.choice([-1., 1.], size=n)
   A    = Y  # Column has correct direction for each sample
   X    = 2 * np.random.rand(n) - 1
   W    = np.full(n, 1 / n)
   f    = 0
   d    = +1
   vMax = n + 1
   
   err  = ErrHinge(X, Y, W)

   FindDist(A, n, Y, W, X, d, vMax, eps0, eps1, eps2, rvs,
            BVae, AWae, AEsi, BVse, AWse, SEsi, tmp)
   errRed, bDist, bSlack = rvs
   
   # Actual error reduction
   nErr   = ErrHinge(X + bDist * A, Y, W)
   actRed = nErr - err

   dTru = np.abs(X-Y).max()
   assert(np.isclose(nErr, 0))
   assert(np.isclose(actRed, errRed))
   assert(np.isclose(dTru, bDist))
   assert(np.isclose(vMax - bDist, bSlack))
# %% Check that values beyond vMax are handled correctly
# Pre-allocate temporary vectors for search
n    = 10
BVae = np.empty(n + 1)
AWae = np.empty(n)
AEsi = np.empty(n, np.intp)

BVse = np.empty(n + 1)
AWse = np.empty(n)
SEsi = np.empty(n, np.intp)

tmp = np.empty(n)

rvs = np.empty(3) # Return values
   
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

   FindDist(A, n, Y, W, X, d, vMax, eps0, eps1, eps2, rvs,
            BVae, AWae, AEsi, BVse, AWse, SEsi, tmp)
   errRed, bDist, bSlack = rvs
   
   # Actual error reduction
   actRed = ErrHinge(X + bDist * A, Y, W) - err

   assert(np.isclose(actRed, errRed))
   assert(np.isclose(vMax, bDist))
   assert(np.isclose(vMax - bDist, bSlack))
# %% Check that optimal value is found when all directions are wrong
# Pre-allocate temporary vectors for search
n    = 10
BVae = np.empty(n + 1)
AWae = np.empty(n)
AEsi = np.empty(n, np.intp)

BVse = np.empty(n + 1)
AWse = np.empty(n)
SEsi = np.empty(n, np.intp)

tmp = np.empty(n)

rvs = np.empty(3) # Return values
   
for i in range(999):
   Y    = np.random.choice([-1., 1.], size=n)
   A    = Y
   X    = 2 * np.random.rand(n) - 1
   W    = np.full(n, 1 / n)
   f    = 0
   d    = -1  # Column has wrong direction for each sample
   vMax = 0.5 
   
   err  = ErrHinge(X, Y, W)

   FindDist(A, n, Y, W, X, d, vMax, eps0, eps1, eps2, rvs,
            BVae, AWae, AEsi, BVse, AWse, SEsi, tmp)
   errRed, bDist, bSlack = rvs
   
   # Actual error reduction
   actRed = ErrHinge(X + bDist * A, Y, W) - err

   assert(np.isclose(actRed, errRed))
   assert(np.isclose(0., bDist))
   assert(np.isclose(vMax - bDist, bSlack))
# %% Compare random instance to brute force solution
# Pre-allocate temporary vectors for search
n    = 10
BVae = np.empty(n + 1)
AWae = np.empty(n)
AEsi = np.empty(n, np.intp)

BVse = np.empty(n + 1)
AWse = np.empty(n)
SEsi = np.empty(n, np.intp)

tmp = np.empty(n)

rvs = np.empty(3) # Return values
   
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

   FindDist(A, n, Y, W, X, d, vMax, eps0, eps1, eps2, rvs,
            BVae, AWae, AEsi, BVse, AWse, SEsi, tmp)
   errRed, bDist, bSlack = rvs
   
   if errRed >= 0:   # No error reduction possible in this dir; mark 0
      bDist  = 0
      errRed = 0
      bSlack = vMax
   
   # Actual error reduction
   nErr   = ErrHinge(X + d * bDist * A, Y, W)
   actRed = nErr - err
   
   # Brute force for optimal solution
   bErr_slow, bDst_slow = SlowCheck(A, Y, X, W, d)
   sErr = bErr_slow - err

   assert(np.isclose(actRed, errRed))
   assert(np.isclose(sErr, actRed))
   assert(np.isclose(vMax - bDist, bSlack))
# %% Print result
print('SUCCESS')
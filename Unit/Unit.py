import  numpy           as     np
from    ICP.PathSearch  import FindDist
from    ICP.Solver      import SignInt8, EPS, DERR_MAX, DEL
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
   A    = (np.arange(n) + np.random.rand(n))[:, None]
   X    = np.zeros(n) # Initial Solution
   dTru = np.random.uniform(1, n - 1)
   Y    = dTru * A[:, 0]
   W    = np.full(n, 1 / n)
   S    = SignInt8(Y)
   B    = S * (Y - X)
   f    = 0
   d    = +1
   vMax = n + 1

   FindDist(A, n, Y, W, S, B, X, f, d, vMax, eps0, eps1, eps2, rvs,
            BVae, AWae, AEsi, BVse, AWse, SEsi, tmp)
   errRed, bDist, bSlack = rvs
   
   # Actual error reduction
   actRed = (S * Y).mean() - np.abs(S * (Y - bDist * A[:, 0])).mean()
   
   assert(np.isclose(actRed, -errRed))
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
   A    = (np.arange(n) + np.random.rand(n))[:, None]
   X    = np.zeros(n) # Initial Solution
   dTru = np.random.uniform(2, n - 1)
   Y    = dTru * A[:, 0]
   W    = np.full(n, 1 / n)
   S    = SignInt8(Y)
   B    = S * (Y - X)
   f    = 0
   d    = +1
   vMax = 1

   FindDist(A, n, Y, W, S, B, X, f, d, vMax, eps0, eps1, eps2, rvs,
            BVae, AWae, AEsi, BVse, AWse, SEsi, tmp)
   errRed, bDist, bSlack = rvs
   
   # Actual error reduction
   actRed = (S * Y).mean() - np.abs(S * (Y - bDist * A[:, 0])).mean()
   
   assert(np.isclose(actRed, -errRed))
   assert(np.isclose(vMax, bDist))
   assert(np.isclose(vMax - bDist, bSlack))
# %% Check that values beyond vMax are handled correctly when incorrect directions are
# present
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
   A     = (np.arange(1, n + 1) + np.random.rand(n))[:, None]
   X     = np.zeros(n) # Initial Solution
   dTru  = np.random.uniform(2, n - 1)
   Y     = dTru * A[:, 0]
   Y[0] *= -1
   W     = np.full(n, 1 / n)
   S     = SignInt8(Y)
   B     = S * (Y - X)
   f     = 0
   d     = +1
   vMax  = 1

   FindDist(A, n, Y, W, S, B, X, f, d, vMax, eps0, eps1, eps2, rvs,
            BVae, AWae, AEsi, BVse, AWse, SEsi, tmp)
   errRed, bDist, bSlack = rvs
   
   # Actual error reduction
   actRed = (S * Y).mean() - np.abs(S * (Y - bDist * A[:, 0])).mean()
   
   assert(np.isclose(actRed, -errRed))
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
   A     = (np.arange(1, n + 1) + np.random.rand(n))[:, None]
   X     = np.zeros(n) # Initial Solution
   dTru  = np.random.uniform(2, n - 1)
   Y     = dTru * A[:, 0]
   W     = np.full(n, 1 / n)
   S     = SignInt8(Y)
   B     = S * (Y - X)
   f     = 0
   d     = -1
   vMax  = 1

   FindDist(A, n, Y, W, S, B, X, f, d, vMax, eps0, eps1, eps2, rvs,
            BVae, AWae, AEsi, BVse, AWse, SEsi, tmp)
   errRed, bDist, bSlack = rvs
   
   # Actual error reduction
   actRed = (S * Y).mean() - np.abs(S * (Y - bDist * A[:, 0])).mean()
   
   assert(np.isclose(actRed, -errRed))
   assert(np.isclose(0, bDist))
   assert(np.isclose(vMax, bSlack))

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel cimport prange
from scipy.linalg.cython_blas cimport sdot
from scipy.special.cython_special cimport psi, gamma
from libc.math cimport log, exp
import ctypes

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void llk_plus_rmse(float* M1, float* M2, float* R,
						int* ix_u, int* ix_i, int nR, int k,
						long double* out, int nthreads, int add_mse, int full_llk) nogil:
	cdef int i
	cdef int one = 1
	cdef float yhat
	cdef long double out1 = 0
	cdef long double out2 = 0
	if add_mse:
		if full_llk:
			for i in prange(nR, schedule='static', num_threads=nthreads):
				yhat = sdot(&k, &M1[ix_u[i] * k], &one, &M2[ix_i[i] * k], &one)
				out1 += R[i]*log(yhat) - log(gamma(R[i] + 1))
				out2 += (R[i] - yhat)**2
		else:
			for i in prange(nR, schedule='static', num_threads=nthreads):
				yhat = sdot(&k, &M1[ix_u[i] * k], &one, &M2[ix_i[i] * k], &one)
				out1 += R[i]*log(yhat)
				out2 += (R[i] - yhat)**2
		out[0] = out1
		out[1] = out2
	else:
		if full_llk:
			for i in prange(nR, schedule='static', num_threads=nthreads):
				out1 += R[i]*log(sdot(&k, &M1[ix_u[i] * k], &one, &M2[ix_i[i] * k], &one)) - log(gamma(R[i] + 1))
			out[0] = out1
		else:
			for i in prange(nR, schedule='static', num_threads=nthreads):
				out1 += R[i]*log(sdot(&k, &M1[ix_u[i] * k], &one, &M2[ix_i[i] * k], &one))
			out[0] = out1
	### Comment: adding += directly to *out triggers compiler optimizations that produce
	### different (and wrong) results across different runs.

def eval_llk(df, np.ndarray[float, ndim=2] Eta, np.ndarray[float, ndim=2] Theta, np.ndarray[float, ndim=2] Eps, k,
			 int nthreads, int add_mse, int full_llk):
	cdef np.ndarray[long double, ndim=1] out = np.empty(2, ctypes.c_longdouble)
	cdef np.ndarray[int, ndim=1] ix_u = df.UserId.values
	cdef np.ndarray[int, ndim=1] ix_i = df.ItemId.values
	cdef np.ndarray[float, ndim=1] Y = df.Count.values
	cdef np.ndarray[float, ndim=2] M2 = Theta + Eps

	llk_plus_rmse(&Eta[0, 0], &M2[0, 0], &Y[0],
				  &ix_u[0], &ix_i[0], df.shape[0],
				  <int> k, &out[0], nthreads, add_mse, full_llk)
	out[0] -= Eta.sum(axis=0).dot(M2.sum(axis=0))
	out[1] = np.sqrt(out[1] / df.shape[0])
	msg = "llk: %d |  rmse: %.4f"
	print msg % (out[0], out[1])
	return out

def fit_ctpf(np.ndarray[float, ndim=2] Theta_shp, np.ndarray[float, ndim=2] Theta_rte,
			 np.ndarray[float, ndim=2] Beta_shp, np.ndarray[float, ndim=2] Beta_rte,
			 np.ndarray[float, ndim=2] Eta_shp, np.ndarray[float, ndim=2] Eta_rte,
			 np.ndarray[float, ndim=2] Eps_shp, np.ndarray[float, ndim=2] Eps_rte,
			 np.ndarray[float, ndim=2] Z, np.ndarray[float, ndim=2] Ya, np.ndarray[float, ndim=2] Yb,
			 df, W, k,
			 a, b, c, d, e, f, g, h,
			 int nthreads):
	cdef int nR = df.shape[0]
	cdef int nW = W.shape[0]
	cdef int nusers = Eta_shp.shape[0]
	cdef int nitems = Theta_shp.shape[0]
	cdef int nwords = Beta_shp.shape[0]
	cdef np.ndarray[float, ndim=1] Warr = W.Count.values
	cdef np.ndarray[int, ndim=1] ix_d_w = W.ItemId.values
	cdef np.ndarray[int, ndim=1] ix_v_w = W.WordId.values
	cdef np.ndarray[float, ndim=1] Rarr = df.Count.values
	cdef np.ndarray[int, ndim=1] ix_u_r = df.UserId.values
	cdef np.ndarray[int, ndim=1] ix_d_r = df.ItemId.values

	cdef np.ndarray[float, ndim=2] Eta = Eta_shp / Eta_rte
	cdef np.ndarray[float, ndim=2] Theta = Theta_shp / Theta_rte
	cdef np.ndarray[float, ndim=2] Eps = Eps_shp / Eps_rte

	## options for llk
	cdef int add_mse = 1
	cdef int full_llk = 1

	## eval llk at the beginning
	# print "random intialization"
	# eval_llk(df, Eta, Theta, Eps, k, nthreads, add_mse, full_llk)
	# print('----')


	for i in range(40):

		## update Z
		update_Z(&Z[0,0], &Theta_shp[0,0], &Theta_rte[0,0], &Beta_shp[0,0], &Beta_rte[0,0],
				 &Warr[0], &ix_d_w[0], &ix_v_w[0], nW, k, nthreads)

		## update ksi
		update_Y(&Ya[0,0], &Yb[0,0], &Eta_shp[0,0], &Eta_rte[0,0],
				   &Theta_shp[0,0], &Theta_rte[0,0], &Eps_shp[0,0], &Eps_rte[0,0],
				   &Rarr[0], &ix_u_r[0], &ix_d_r[0], nR, k, nthreads)

		## update theta
		Theta_shp[:,:] = c
		update_Theta_shp(&Theta_shp[0,0], &Z[0,0], &Ya[0,0],
					     &ix_d_w[0], &ix_d_r[0], nW, nR, k, 0, nthreads)
		Theta_rte[:,:] = d + (Beta_shp/Beta_rte).sum(axis=0, keepdims=True) + Eta.sum(axis=0, keepdims=True)
		## FIXME: for some reason, Theta_shp and Theta_rte get wrong results, but can be somehow fixed by adding this:
		Theta_shp = Theta_shp + np.zeros((Theta_shp.shape[0], Theta_shp.shape[1]), dtype='float32')
		Theta_rte = Theta_rte + np.zeros((Theta_rte.shape[0], Theta_rte.shape[1]), dtype='float32')
		Theta[:,:] = Theta_shp / Theta_rte

		# print ('updated theta')
		# eval_llk(df, Eta, Theta, Eps, k)

		## update beta
		Beta_shp[:,:] = a
		update_Beta_shp(&Beta_shp[0,0], &Z[0,0], &ix_v_w[0], nW, k, nthreads)
		Beta_rte[:,:] = b + Theta.sum(axis=0, keepdims=True)

		# update eta
		Eta_shp[:,:] = e
		update_Eta_shp(&Eta_shp[0,0], &Ya[0,0], &Yb[0,0], &ix_u_r[0], nR, k, nthreads)
		Eta_rte[:,:] = f + (Theta + Eps).sum(axis=0, keepdims=True)
		Eta[:,:] = Eta_shp / Eta_rte

		# print ('updated beta and eta')
		# eval_llk(df, Eta, Theta, Eps, k)

		## update epsilon
		Eps_shp[:,:] = g
		update_Eps_shp(&Eps_shp[0,0], &Yb[0,0], &ix_d_r[0], nR, k, 0, nthreads)
		Eps_rte[:,:] = h + Eta.sum(axis=0, keepdims=True)
		Eps[:,:] = Eps_shp / Eps_rte

		# print ('updated eps')
		# eval_llk(df, Eta, Theta, Eps, k)
		# print("-----")
		if ((i+1)%10) == 0:
			eval_llk(df, Eta, Theta, Eps, k, nthreads, add_mse, full_llk)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void update_Z(float* Z, float* Theta_shp, float* Theta_rte,
					 float* Beta_shp, float* Beta_rte, float* W,
					 int* ix_d, int* ix_v, int nW, int K, int nthreads) nogil:
	cdef int i, k
	cdef int st_ix_z, st_ix_theta, st_ix_beta
	cdef float sumrow, maxval

	for i in prange(nW, schedule='static', num_threads=nthreads):
		st_ix_z = i * K
		st_ix_theta = ix_d[i] * K
		st_ix_beta = ix_v[i] * K

		sumrow = 0
		maxval =  - 10**10
		for k in range(K):
			Z[st_ix_z + k] = psi(Theta_shp[st_ix_theta + k]) - log(Theta_rte[st_ix_theta + k]) + psi(Beta_shp[st_ix_beta + k]) - log(Beta_rte[k])
			if Z[st_ix_z + k] > maxval:
				maxval = Z[st_ix_z + k]
		for k in range(K):
			Z[st_ix_z + k] = exp(Z[st_ix_z + k] - maxval)
			sumrow += Z[st_ix_z + k]
		for k in range(K):
			Z[st_ix_z + k] *= W[i] / sumrow

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void update_Y(float* Ya, float* Yb, float* Eta_shp, float* Eta_rte, float* Theta_shp, float* Theta_rte,
				   float* Eps_shp, float* Eps_rte, float* R, int* ix_u_r, int* ix_d_r, int nR, int K, int nthreads) nogil:
	cdef int i, k
	cdef int st_ix_y, st_ix_u, st_ix_d, ix_2
	cdef float E_eta, sumrow, maxval

	for i in prange(nR, schedule='static', num_threads=nthreads):
		st_ix_u = ix_u_r[i] * K
		st_ix_d = ix_d_r[i] * K
		st_ix_y = i * K

		sumrow = 0
		maxval = - 10**10
		for k in range(K):
			E_eta = psi(Eta_shp[st_ix_u + k]) - log(Eta_rte[st_ix_u + k])
			ix_2 = st_ix_d + k
			Ya[st_ix_y + k] = E_eta + psi(Theta_shp[ix_2]) - log(Theta_rte[ix_2])
			Yb[st_ix_y + k] = E_eta + psi(Eps_shp[ix_2]) - log(Eps_rte[ix_2])
			# Yb[st_ix_y + k] = E_eta + psi(Eps_shp[ix_2]) - log(Eps_rte[k])
			if Ya[st_ix_y + k] > maxval:
				maxval = Ya[st_ix_y + k]
			if Yb[st_ix_y + k] > maxval:
				maxval = Yb[st_ix_y + k]
		for k in range(K):
			Ya[st_ix_y + k] = exp(Ya[st_ix_y + k] - maxval)
			Yb[st_ix_y + k] = exp(Yb[st_ix_y + k] - maxval)
			sumrow += Ya[st_ix_y + k] + Yb[st_ix_y + k]
		for k in range(K):
			Ya[st_ix_y + k] *= R[i] / sumrow
			Yb[st_ix_y + k] *= R[i] / sumrow


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void update_Theta_shp(float* Theta_shp, float* Z, float* Ya,
					   int* ix_d_w, int* ix_d_r, int nW, int nR, int K,
					   int allow_inconsistent, int nthreads) nogil:
	cdef int i, j, k
	cdef int st_ix_theta, st_ix_z, st_ix_y
	if allow_inconsistent:
		for i in prange(nW, schedule='static', num_threads=nthreads):
			st_ix_theta = ix_d_w[i] * K
			st_ix_z = i * K
			for k in range(K):
				Theta_shp[st_ix_theta + k] += Z[st_ix_z + k]
		for j in prange(nR, schedule='static', num_threads=nthreads):
			st_ix_theta = ix_d_r[i] * K
			st_ix_y = j * K
			for k in range(K):
				Theta_shp[st_ix_theta + k] += Ya[st_ix_y + k]
	else:
		for i in range(nW):
			st_ix_theta = ix_d_w[i] * K
			st_ix_z = i * K
			for k in range(K):
				Theta_shp[st_ix_theta + k] += Z[st_ix_z + k]
		for j in range(nR):
			st_ix_theta = ix_d_r[i] * K
			st_ix_y = j * K
			for k in range(K):
				Theta_shp[st_ix_theta + k] += Ya[st_ix_y + k]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void update_Beta_shp(float* Beta_shp, float* Z, int* ix_v, int nW, int K, int nthreads) nogil:
	cdef int i, k
	cdef int st_ix_beta, st_ix_Z
	for i in prange(nW, schedule='static', num_threads=nthreads):
		st_ix_Z = i * K
		st_ix_beta = ix_v[i] * K
		for k in range(K):
			Beta_shp[st_ix_beta + k] += Z[st_ix_Z + k]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void update_Eta_shp(float* Eta_shp, float* Ya, float* Yb,
						 int* ix_u, int nR, int K, int nthreads) nogil:
	cdef int i, k
	cdef int st_ix_ksi, st_ix_eta
	for i in prange(nR, schedule='static', num_threads=nthreads):
		st_ix_eta = ix_u[i] * K
		st_ix_ksi = i * K
		for k in range(K):
			Eta_shp[st_ix_eta + k] += Ya[st_ix_ksi + k] + Yb[st_ix_ksi + k]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void update_Eps_shp(float* Eps_shp, float* Yb, int* ix_d, int nR, int K,
						 int allow_inconsistent, int nthreads) nogil:
	cdef int i, k
	cdef int st_ix_eps, st_ix_ksi
	if allow_inconsistent:
		for i in prange(nR, schedule='static', num_threads=nthreads):
			st_ix_eps = ix_d[i] * K
			st_ix_ksi = i * K
			for k in range(K):
				Eps_shp[st_ix_eps + k] += Yb[st_ix_ksi + k]
	else:
		for i in range(nR):
			st_ix_eps = ix_d[i] * K
			st_ix_ksi = i * K
			for k in range(K):
				Eps_shp[st_ix_eps + k] += Yb[st_ix_ksi + k]


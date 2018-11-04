import numpy as np
cimport numpy as np
cimport cython
from cython.parallel cimport prange
from scipy.linalg.cython_blas cimport sdot
from scipy.special.cython_special cimport psi, gamma
from libc.math cimport log, exp
import ctypes
from hpfrec import cython_loops
import time

### Main function
#################
def fit_ctpf(np.ndarray[float, ndim=2] Theta_shp, np.ndarray[float, ndim=2] Theta_rte,
			 np.ndarray[float, ndim=2] Beta_shp, np.ndarray[float, ndim=2] Beta_rte,
			 np.ndarray[float, ndim=2] Eta_shp, np.ndarray[float, ndim=2] Eta_rte,
			 np.ndarray[float, ndim=2] Eps_shp, np.ndarray[float, ndim=2] Eps_rte,
			 np.ndarray[float, ndim=2] Omega_shp, np.ndarray[float, ndim=2] Omega_rte,
			 np.ndarray[float, ndim=2] Kappa_shp, np.ndarray[float, ndim=2] Kappa_rte,
			 np.ndarray[float, ndim=2] Theta, np.ndarray[float, ndim=2] Eta,
			 np.ndarray[float, ndim=2] Eps, np.ndarray[float, ndim=2] Omega,
			 user_df, has_user_df,
			 df, W, size_t k, step_size, has_step_size, int sum_exp_trick,
			 float a, float b, float c, float d, float e, float f, float g, float h,
			 int nthreads, int maxiter, int miniter, int check_every,
			 stop_crit, stop_thr, verbose, save_folder,
			 int allow_inconsistent, int has_valset, int full_llk,
			 val_df
			 ):
	# TODO: implement stochastic variational inference taking CSR data and small Z-X-Y matrices (take from hpfrec)
	# TODO: for the SVI model with user and item info, alternate between epochs of users and items
	cdef size_t nR = df.shape[0]
	cdef size_t nW = W.shape[0]
	cdef size_t nusers = Eta_shp.shape[0]
	cdef size_t nitems = Theta_shp.shape[0]
	cdef size_t nwords = Beta_shp.shape[0]
	cdef np.ndarray[float, ndim=1] Warr = W.Count.values
	cdef np.ndarray[size_t, ndim=1] ix_d_w = W.ItemId.values
	cdef np.ndarray[size_t, ndim=1] ix_v_w = W.WordId.values
	cdef np.ndarray[float, ndim=1] Rarr = df.Count.values
	cdef np.ndarray[size_t, ndim=1] ix_u_r = df.UserId.values
	cdef np.ndarray[size_t, ndim=1] ix_d_r = df.ItemId.values

	cdef np.ndarray[float, ndim=1] Rval = val_df.Count.values
	cdef np.ndarray[size_t, ndim=1] ix_u_val = val_df.UserId.values
	cdef np.ndarray[size_t, ndim=1] ix_d_val = val_df.ItemId.values
	cdef size_t nRv = val_df.shape[0]

	cdef np.ndarray[float, ndim=1] Qarr
	cdef np.ndarray[size_t, ndim=1] ix_u_q, ix_a_q
	cdef size_t nQ, nattr
	if has_user_df:
		Qarr = user_df.Count.values
		ix_u_q = user_df.UserId.values
		ix_a_q = user_df.AttributeId.values
		nQ = user_df.shape[0]
		nattr = Kappa_shp.shape[0]

	cdef np.ndarray[float, ndim=2] Theta_shp_prev, Theta_rte_prev, Beta_shp_prev, Beta_rte_prev
	cdef np.ndarray[float, ndim=2] Theta_prev
	if stop_crit == 'diff-norm':
		Theta_prev = Theta.copy()
	else:
		Theta_prev = np.empty((0,0), dtype='float32')
	cdef long double last_crit = - (10**37)
	cdef np.ndarray[long double, ndim=1] errs = np.zeros(2, dtype=ctypes.c_longdouble)

	if verbose>0:
		print "Allocating intermediate matrices..."
	cdef np.ndarray[float, ndim=2] Z = np.empty((nW, k), dtype='float32')
	cdef np.ndarray[float, ndim=2] Ya = np.empty((nR, k), dtype='float32')
	cdef np.ndarray[float, ndim=2] Yb = np.empty((nR, k), dtype='float32')
	cdef np.ndarray[float, ndim=2] Yc, Yd, X
	if has_user_df:
		Yc = np.empty((nR, k), dtype='float32')
		Yd = np.empty((nR, k), dtype='float32')
		X = np.empty((nQ, k), dtype='float32')

	if verbose>0:
		print "Initializing optimization procedure..."
	cdef double st_time = time.time()

	for i in range(maxiter):

		## regular model without user side info
		if not has_user_df:
			## update Z (phi)
			update_Z(&Z[0,0], &Theta_shp[0,0], &Theta_rte[0,0], &Beta_shp[0,0], &Beta_rte[0,0],
					 &Warr[0], &ix_d_w[0], &ix_v_w[0], sum_exp_trick, nW, k, nthreads)

			## update Y (ksi)
			update_Y(&Ya[0,0], &Yb[0,0], &Eta_shp[0,0], &Eta_rte[0,0],
					   &Theta_shp[0,0], &Theta_rte[0,0], &Eps_shp[0,0], &Eps_rte[0,0],
					   &Rarr[0], &ix_u_r[0], &ix_d_r[0], sum_exp_trick, nR, k, nthreads)

			## update theta
			if has_step_size:
				Theta_shp_prev = Theta_shp.copy()
			Theta_shp[:,:] = c
			update_Theta_shp(&Theta_shp[0,0], &Z[0,0], &Ya[0,0],
						     &ix_d_w[0], &ix_d_r[0], nW, nR, k, allow_inconsistent, nthreads)
			if has_step_size:
				Theta_shp[:,:] = step_size(i) * Theta_shp + (1 - step_size(i)) * Theta_shp_prev
				Theta_rte[:,:] = (1 - step_size(i))*Theta_rte + step_size(i) * (d + \
									(Beta_shp/Beta_rte).sum(axis=0, keepdims=True) + \
									Eta.sum(axis=0, keepdims=True))
			else:
				Theta_rte[:,:] = d + (Beta_shp/Beta_rte).sum(axis=0, keepdims=True) + Eta.sum(axis=0, keepdims=True)
			## FIXME: for some reason, Theta_shp and Theta_rte get wrong results, but can be somehow fixed by adding this:
			Theta_shp = Theta_shp + np.zeros((Theta_shp.shape[0], Theta_shp.shape[1]), dtype='float32')
			Theta_rte = Theta_rte + np.zeros((Theta_rte.shape[0], Theta_rte.shape[1]), dtype='float32')
			Theta[:,:] = Theta_shp / Theta_rte

			## update beta
			if has_step_size:
				Beta_shp_prev = Beta_shp.copy()
			Beta_shp[:,:] = a
			update_Beta_shp(&Beta_shp[0,0], &Z[0,0], &ix_v_w[0], nW, k, nthreads)
			if has_step_size:
				Beta_shp[:,:] = step_size(i) * Beta_shp + (1 - step_size(i)) * Beta_shp_prev
				Beta_rte[:,:] = step_size(i) * (b + Theta.sum(axis=0, keepdims=True)) + (1 - step_size(i)) * Beta_rte
			else:
				Beta_rte[:,:] = b + Theta.sum(axis=0, keepdims=True)

			# update eta
			Eta_shp[:,:] = e
			update_Eta_shp(&Eta_shp[0,0], &Ya[0,0], &Yb[0,0], &ix_u_r[0], nR, k, nthreads)
			Eta_rte[:,:] = f + (Theta + Eps).sum(axis=0, keepdims=True)
			Eta[:,:] = Eta_shp / Eta_rte

			## update epsilon
			Eps_shp[:,:] = g
			update_Eps_shp(&Eps_shp[0,0], &Yb[0,0], &ix_d_r[0], nR, k, allow_inconsistent, nthreads)
			Eps_rte[:,:] = h + Eta.sum(axis=0, keepdims=True)
			Eps[:,:] = Eps_shp / Eps_rte

		## model with user side info
		else:
			## Z:= softmax( psi(Theta_shp) - log(Theta_rte) + psi(Beta_shp) - log(Beta_rte) )
			update_Z(&Z[0,0], &Theta_shp[0,0], &Theta_rte[0,0], &Beta_shp[0,0], &Beta_rte[0,0],
					 &Warr[0], &ix_d_w[0], &ix_v_w[0], 1, nW, k, nthreads)

			## X := softmax( psi(Omega_shp) - log(Omega_rte) + psi(Kappa_shp) - log(Kappa_rte) )
			update_Z(&X[0,0], &Omega_shp[0,0], &Omega_rte[0,0], &Kappa_shp[0,0], &Kappa_rte[0,0],
					 &Qarr[0], &ix_u_q[0], &ix_a_q[0], 1, nQ, k, nthreads)

			## Ya := softmax( psi(Omega_shp) - log(Omega_shp) + psi(Theta_shp) - log(Theta_rte) );
			## Yb := softmax( psi(Eta_shp) - log(Eta_shp) + psi(Theta_shp) - log(Theta_rte) );
			## Yc := softmax( psi(Omega_shp) - log(Omega_shp) + psi(Eps_shp) - log(Eps_rte) );
			## Yd := softmax( psi(Eta_shp) - log(Eta_rte) + psi(Eps_shp) - log(Eps_rte) );
			update_Y(&Ya[0,0], &Yc[0,0], &Omega_shp[0,0], &Omega_rte[0,0],
					   &Theta_shp[0,0], &Theta_rte[0,0], &Eps_shp[0,0], &Eps_rte[0,0],
					   &Rarr[0], &ix_u_r[0], &ix_d_r[0], 1, nR, k, nthreads)
			update_Y(&Yb[0,0], &Yd[0,0], &Eta_shp[0,0], &Eta_rte[0,0],
					   &Theta_shp[0,0], &Theta_rte[0,0], &Eps_shp[0,0], &Eps_rte[0,0],
					   &Rarr[0], &ix_u_r[0], &ix_d_r[0], 1, nR, k, nthreads)

			## Theta_shp := c + sum_w(Z) + sum_u(Ya + Yb)
			## Theta_rte := d + sum_w(Beta) + sum_u(Omega + Eta)
			if has_step_size:
				Theta_shp_prev = Theta_shp.copy()
			Theta_shp[:,:] = c
			update_Theta_shp_wuser(&Theta_shp[0,0], &Z[0,0], &Ya[0,0], &Yb[0,0],
						     &ix_d_w[0], &ix_d_r[0], nW, nR, k, allow_inconsistent, nthreads)
			if has_step_size:
				Theta_shp[:,:] = step_size(i) * Theta_shp + (1 - step_size(i)) * Theta_shp_prev
				Theta_rte[:,:] = (1 - step_size(i))*Theta_rte + step_size(i) * (d + \
									(Beta_shp/Beta_rte).sum(axis=0, keepdims=True) + \
									(Omega + Eta).sum(axis=0, keepdims=True))
			else:
				Theta_rte[:,:] = d + (Beta_shp/Beta_rte).sum(axis=0, keepdims=True) + (Omega + Eta).sum(axis=0, keepdims=True)
			## FIXME: for some reason, Theta_shp and Theta_rte get wrong results, but can be somehow fixed by adding this:
			Theta_shp = Theta_shp + np.zeros((Theta_shp.shape[0], Theta_shp.shape[1]), dtype='float32')
			Theta_rte = Theta_rte + np.zeros((Theta_rte.shape[0], Theta_rte.shape[1]), dtype='float32')
			Theta[:,:] = Theta_shp / Theta_rte

			## Beta_shp := a + sum_i(Z)
			## Beta_rte := b + sum_i(Theta)
			if has_step_size:
				Beta_shp_prev = Beta_shp.copy()
			Beta_shp[:,:] = a
			update_Beta_shp(&Beta_shp[0,0], &Z[0,0], &ix_v_w[0], nW, k, nthreads)
			if has_step_size:
				Beta_shp[:,:] = step_size(i) * Beta_shp + (1 - step_size(i)) * Beta_shp_prev
				Beta_rte[:,:] = step_size(i) * (b + Theta.sum(axis=0, keepdims=True)) + (1 - step_size(i)) * Beta_rte
			else:
				Beta_rte[:,:] = b + Theta.sum(axis=0, keepdims=True)

			## Omega_shp := c + sum_a(X) + sum_i(Ya + Yc)
			## Omega_rte := d + sum_a(Kappa) + sum_i(Theta + Eps)
			Omega_shp[:,:] = c
			update_Theta_shp_wuser(&Omega_shp[0,0], &X[0,0], &Ya[0,0], &Yc[0,0],
						     &ix_u_q[0], &ix_u_r[0], nQ, nR, k, allow_inconsistent, nthreads)
			Omega_rte[:,:] = d + (Kappa_shp/Kappa_rte).sum(axis=0, keepdims=True) + (Theta + Eps).sum(axis=0, keepdims=True)
			## FIXME: for some reason, Omega_shp and Omega_rte get wrong results, but can be somehow fixed by adding this:
			Omega_shp = Omega_shp + np.zeros((Omega_shp.shape[0], Omega_shp.shape[1]), dtype='float32')
			Omega_rte = Omega_rte + np.zeros((Omega_rte.shape[0], Omega_rte.shape[1]), dtype='float32')
			Omega[:,:] = Omega_shp / Omega_rte

			## Kappa_shp := a + sum_u(X)
			## Kappa_rte := b + sum_u(Omega)
			if has_step_size:
				Kappa_shp_prev = Kappa_shp.copy()
			Kappa_shp[:,:] = a
			update_Beta_shp(&Kappa_shp[0,0], &X[0,0], &ix_a_q[0], nQ, k, nthreads)
			if has_step_size:
				Kappa_shp[:,:] = step_size(i) * Kappa_shp + (1 - step_size(i)) * Kappa_shp_prev
				Kappa_rte[:,:] = step_size(i) * (b + Omega.sum(axis=0, keepdims=True)) + (1 - step_size(i)) * Kappa_rte
			else:
				Kappa_rte[:,:] = b + Omega.sum(axis=0, keepdims=True)

			## Eps_shp := g + sum_u(Yc + Yd)
			## Eps_rte := h + sum_u(Omega + Eta)
			Eps_shp[:,:] = g
			update_Eta_shp(&Eps_shp[0,0], &Yc[0,0], &Yd[0,0], &ix_d_r[0], nR, k, nthreads)
			Eps_rte[:,:] = h + (Omega + Eta).sum(axis=0, keepdims=True)
			Eps[:,:] = Eps_shp / Eps_rte

			## Eta_shp := e + sum_i(Yb + Yd)
			## Eta_rte := f + sum_i(Theta + Eps)
			Eta_shp[:,:] = e
			update_Eta_shp(&Eta_shp[0,0], &Yb[0,0], &Yd[0,0], &ix_u_r[0], nR, k, nthreads)
			Eta_rte[:,:] = f + (Theta + Eps).sum(axis=0, keepdims=True)
			Eta[:,:] = Eta_shp / Eta_rte

		## assessing convergence
		if check_every>0:
			if ((i+1) % check_every) == 0:

				has_converged, last_crit = assess_convergence(
					i, check_every, stop_crit, last_crit, stop_thr,
					Theta, Theta_prev,
					Eta, Eps,
					nR,
					Rarr, ix_u_r, ix_d_r, nRv,
					Rval, ix_u_val, ix_d_val,
					errs, k, nthreads, verbose, full_llk, has_valset
					)

				if has_converged and (i > miniter):
					if (stop_crit == 'diff-norm') and has_step_size:
						if step_size(i) <= 1e-2:
							continue
						else:
							break
					else:
						break

	## last metrics once it finishes optimizing
	cython_loops.eval_after_term(
					stop_crit, verbose, nthreads, full_llk, k, nR, nRv, has_valset,
					Eta, Theta+Eps, errs,
					Rarr, ix_u_r, ix_d_r,
					Rval, ix_u_val, ix_d_val
					)

	cdef double end_tm = (time.time()-st_time)/60
	if verbose:
		cython_loops.print_final_msg(i+1, <long long> errs[0], <double> errs[1], end_tm)

	if save_folder != "":
		cython_loops.save_parameters(verbose, save_folder,
						["Theta_shp", "Theta_rte", "Beta_shp", "Beta_rte", "Eta_shp",
						"Eta_rte", "Eps_shp", "Eps_rte"],
						[Theta_shp, Theta_rte, Beta_shp, Beta_rte, Eta_shp,
						Eta_rte, Eps_shp, Eps_rte])

	return i

### Helpers
###########
def assess_convergence(int i, check_every, stop_crit, last_crit, stop_thr,
					   np.ndarray[float, ndim=2] Theta, np.ndarray[float, ndim=2] Theta_prev,
					   np.ndarray[float, ndim=2] Eta, np.ndarray[float, ndim=2] Eps,
					   size_t nY,
					   np.ndarray[float, ndim=1] Y, np.ndarray[size_t, ndim=1] ix_u, np.ndarray[size_t, ndim=1] ix_i, size_t nYv,
					   np.ndarray[float, ndim=1] Yval, np.ndarray[size_t, ndim=1] ix_u_val, np.ndarray[size_t, ndim=1] ix_i_val,
					   np.ndarray[long double, ndim=1] errs, size_t k, int nthreads, int verbose, int full_llk, has_valset):

	cdef np.ndarray[float, ndim=2] M2

	if stop_crit == 'diff-norm':
		last_crit = np.linalg.norm(Theta - Theta_prev)
		if verbose:
			cython_loops.print_norm_diff(i+1, check_every, <float> last_crit)
		if last_crit < stop_thr:
			return True, last_crit
		Theta_prev[:,:] = Theta.copy()

	else:

		M2 = Theta + Eps
		if has_valset:
			llk_plus_rmse(&Eta[0,0], &M2[0,0], &Yval[0],
						  &ix_u_val[0], &ix_i_val[0], nYv, k,
						  &errs[0], nthreads, verbose, full_llk)
			errs[0] -= Eta[ix_u_val].sum(axis=0).dot(M2[ix_i_val].sum(axis=0))
			errs[1] = np.sqrt(errs[1]/nYv)
		else:
			llk_plus_rmse(&Eta[0,0], &M2[0,0], &Y[0],
						  &ix_u[0], &ix_i[0], nY, k,
						  &errs[0], nthreads, verbose, full_llk)
			errs[0] -= Eta.sum(axis=0).dot(M2.sum(axis=0))
			errs[1] = np.sqrt(errs[1]/nY)

		if verbose:
			cython_loops.print_llk_iter(<int> (i+1), <long long> errs[0], <double> errs[1], has_valset)

		if stop_crit != 'maxiter':
			if (i+1) == check_every:
				last_crit = errs[0]
			else:
				if (1 - errs[0]/last_crit) <= stop_thr:
					return True, last_crit
				last_crit = errs[0]
	
	return False, last_crit

### Functions for updating without refitting
############################################
def calc_item_factors(W, size_t nitems, int maxiter, size_t k, stop_thr, random_seed, int nthreads,
					  float a, float b, float c, float d,
					  np.ndarray[float, ndim=2] Theta_rte,
					  np.ndarray[float, ndim=2] Beta_shp, np.ndarray[float, ndim=2] Beta_rte):
	cdef np.ndarray[float, ndim=1] Warr = W.Count.values
	cdef np.ndarray[size_t, ndim=1] ix_i_w = W.ItemId.values
	cdef np.ndarray[size_t, ndim=1] ix_v_w = W.WordId.values
	cdef size_t nW = W.shape[0]

	if random_seed is not None:
		np.random.seed(random_seed)

	cdef np.ndarray[float, ndim=2] Theta_shp = (a * 2*np.random.beta(20, 20, size=(nitems, k))).astype('float32')
	cdef np.ndarray[float, ndim=2] Theta_prev = Theta_shp.copy()
	cdef np.ndarray[float, ndim=2] Z = np.empty((nW, k), dtype='float32')
	cdef np.ndarray[float, ndim=2] Zconst = np.empty((nW, k), dtype='float32')
	update_Z_const_pred(&Zconst[0,0], &Theta_rte[0,0], &Beta_shp[0,0], &Beta_rte[0,0],
						&ix_i_w[0], &ix_v_w[0], nW, k, nthreads)

	for i in range(maxiter):
		update_Z_var_pred(&Z[0,0], &Zconst[0,0], &Warr[0], &Theta_shp[0,0], &ix_i_w[0],
						  nW, k, nthreads)
		Theta_shp[:,:] = c
		update_Theta_shp_pred(&Theta_shp[0,0], &Z[0,0], &ix_i_w[0],
							  nW, k, nthreads)

		if np.linalg.norm(Theta_shp - Theta_prev) <= stop_thr:
			break
		else:
			Theta_prev[:,:] = Theta_shp.copy()

	return Theta_shp, Z

def calc_user_factors(df, size_t nusers, int maxiter, size_t k, stop_thr, random_seed, int nthreads,
					  float e, np.ndarray[float, ndim=2] Eta_rte,
					  np.ndarray[float, ndim=2] Theta_shp, np.ndarray[float, ndim=2] Theta_rte,
					  np.ndarray[float, ndim=2] Eps_shp, np.ndarray[float, ndim=2] Eps_rte):
	cdef size_t nR = df.shape[0]
	cdef np.ndarray[float, ndim=1] Rarr = df.Count.values
	cdef np.ndarray[size_t, ndim=1] ix_u_r = df.UserId.values
	cdef np.ndarray[size_t, ndim=1] ix_i_r = df.ItemId.values
	cdef np.ndarray[float, ndim=2] Ya = np.empty((nR, k), dtype='float32')
	cdef np.ndarray[float, ndim=2] Ya_const = np.empty((nR, k), dtype='float32')
	cdef np.ndarray[float, ndim=2] Yb = np.empty((nR, k), dtype='float32')
	cdef np.ndarray[float, ndim=2] Yb_const = np.empty((nR, k), dtype='float32')

	if random_seed is not None:
		np.random.seed(random_seed)
	cdef np.ndarray[float, ndim=2] Eta_shp = e * 2*np.random.beta(20, 20, size=(nusers, k)).astype('float32')
	cdef np.ndarray[float, ndim=2] Eta_prev = Eta_shp.copy()

	## reusing the same functions for items with different parameters only
	update_Z_const_pred(&Ya_const[0,0], &Eta_rte[0,0], &Theta_shp[0,0], &Theta_rte[0,0],
						&ix_u_r[0], &ix_i_r[0], nR, k, nthreads)
	update_Z_const_pred(&Yb_const[0,0], &Eta_rte[0,0], &Eps_shp[0,0], &Eps_rte[0,0],
						&ix_u_r[0], &ix_i_r[0], nR, k, nthreads)

	for i in range(maxiter):
		update_Z_var_pred(&Ya[0,0], &Ya_const[0,0], &Rarr[0], &Eta_shp[0,0], &ix_u_r[0],
						  nR, k, nthreads)
		update_Z_var_pred(&Yb[0,0], &Yb_const[0,0], &Rarr[0], &Eta_shp[0,0], &ix_u_r[0],
						  nR, k, nthreads)
		
		Eta_shp[:,:] = e
		update_Eta_shp(&Eta_shp[0,0], &Ya[0,0], &Yb[0,0], &ix_u_r[0], nR, k, nthreads)

		if np.linalg.norm(Eta_prev - Eta_shp) <= stop_thr:
			break
		else:
			Eta_prev = Eta_shp.copy()

	return Eta_shp

def calc_user_factors_full(df, user_df, size_t nusers, int maxiter, size_t k, stop_thr, random_seed, int nthreads,
					  float c, float e, np.ndarray[float, ndim=2] Omega_rte, np.ndarray[float, ndim=2] Eta_rte,
					  np.ndarray[float, ndim=2] Theta_shp, np.ndarray[float, ndim=2] Theta_rte,
					  np.ndarray[float, ndim=2] Eps_shp, np.ndarray[float, ndim=2] Eps_rte,
					  np.ndarray[float, ndim=2] Kappa_shp, np.ndarray[float, ndim=2] Kappa_rte):
	cdef size_t nR = df.shape[0]
	cdef np.ndarray[float, ndim=1] Rarr = df.Count.values
	cdef np.ndarray[size_t, ndim=1] ix_u_r = df.UserId.values
	cdef np.ndarray[size_t, ndim=1] ix_i_r = df.ItemId.values

	cdef np.ndarray[float, ndim=2] Ya = np.empty((nR, k), dtype='float32')
	cdef np.ndarray[float, ndim=2] Ya_const = np.empty((nR, k), dtype='float32')
	cdef np.ndarray[float, ndim=2] Yb = np.empty((nR, k), dtype='float32')
	cdef np.ndarray[float, ndim=2] Yb_const = np.empty((nR, k), dtype='float32')
	cdef np.ndarray[float, ndim=2] Yc = np.empty((nR, k), dtype='float32')
	cdef np.ndarray[float, ndim=2] Yc_const = np.empty((nR, k), dtype='float32')
	cdef np.ndarray[float, ndim=2] Yd = np.empty((nR, k), dtype='float32')
	cdef np.ndarray[float, ndim=2] Yd_const = np.empty((nR, k), dtype='float32')

	cdef size_t nQ = user_df.shape[0]
	cdef np.ndarray[float, ndim=1] Qarr = user_df.Count.values
	cdef np.ndarray[size_t, ndim=1] ix_u_q = user_df.UserId.values
	cdef np.ndarray[size_t, ndim=1] ix_a_q = user_df.AttributeId.values
	cdef np.ndarray[float, ndim=2] X = np.empty((nQ, k), dtype='float32')

	if random_seed is not None:
		np.random.seed(random_seed)
	cdef np.ndarray[float, ndim=2] Eta_shp = e * 2*np.random.beta(20, 20, size=(nusers, k)).astype('float32')
	cdef np.ndarray[float, ndim=2] Eta_prev = Eta_shp.copy()
	cdef np.ndarray[float, ndim=2] Omega_shp = c * 2*np.random.beta(20, 20, size=(nusers, k)).astype('float32')

	## reusing the same functions for items with different parameters only
	update_Z_const_pred(&Ya_const[0,0], &Omega_rte[0,0], &Theta_shp[0,0], &Theta_rte[0,0],
						&ix_u_r[0], &ix_i_r[0], nR, k, nthreads)
	update_Z_const_pred(&Yb_const[0,0], &Eta_rte[0,0], &Theta_shp[0,0], &Theta_rte[0,0],
						&ix_u_r[0], &ix_i_r[0], nR, k, nthreads)
	update_Z_const_pred(&Yc_const[0,0], &Omega_rte[0,0], &Eps_shp[0,0], &Eps_rte[0,0],
						&ix_u_r[0], &ix_i_r[0], nR, k, nthreads)
	update_Z_const_pred(&Yd_const[0,0], &Eta_rte[0,0], &Eps_shp[0,0], &Eps_rte[0,0],
						&ix_u_r[0], &ix_i_r[0], nR, k, nthreads)

	for i in range(maxiter):
		update_Z(&X[0,0], &Omega_shp[0,0], &Omega_rte[0,0], &Kappa_shp[0,0], &Kappa_rte[0,0],
            &Qarr[0], &ix_u_q[0], &ix_a_q[0], 1, nQ, k, nthreads)

		update_Z_var_pred(&Ya[0,0], &Ya_const[0,0], &Rarr[0], &Omega_shp[0,0], &ix_u_r[0],
						  nR, k, nthreads)
		update_Z_var_pred(&Yb[0,0], &Yb_const[0,0], &Rarr[0], &Eta_shp[0,0], &ix_u_r[0],
						  nR, k, nthreads)
		update_Z_var_pred(&Yc[0,0], &Yc_const[0,0], &Rarr[0], &Omega_shp[0,0], &ix_u_r[0],
						  nR, k, nthreads)
		update_Z_var_pred(&Yd[0,0], &Yd_const[0,0], &Rarr[0], &Eta_shp[0,0], &ix_u_r[0],
						  nR, k, nthreads)
		
		Eta_shp[:,:] = e
		update_Eta_shp(&Eta_shp[0,0], &Yb[0,0], &Yd[0,0], &ix_u_r[0], nR, k, nthreads)

		Omega_shp[:,:] = c
		update_Beta_shp(&Omega_shp[0,0], &X[0,0], &ix_u_q[0], nQ, k, nthreads)
		update_Eta_shp(&Omega_shp[0,0], &Ya[0,0], &Yc[0,0], &ix_u_r[0], nR, k, nthreads)

		if np.linalg.norm(Eta_prev - Eta_shp) <= stop_thr:
			break
		else:
			Eta_prev = Eta_shp.copy()

	return Omega_shp, Eta_shp

### Fast and parallel C functions
#################################
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void update_Z(float* Z, float* Theta_shp, float* Theta_rte,
					 float* Beta_shp, float* Beta_rte, float* W,
					 size_t* ix_d, size_t* ix_v, int sum_exp_trick,
					 size_t nW, size_t K, int nthreads) nogil:
	cdef size_t i, k
	cdef size_t st_ix_z, st_ix_theta, st_ix_beta
	cdef float sumrow, maxval

	if sum_exp_trick:
		for i in prange(nW, schedule='static', num_threads=nthreads):
			st_ix_z = i * K
			st_ix_theta = ix_d[i] * K
			st_ix_beta = ix_v[i] * K
			sumrow = 0
			maxval =  - 10**1
			for k in range(K):
				Z[st_ix_z + k] = psi(Theta_shp[st_ix_theta + k]) - log(Theta_rte[k]) + psi(Beta_shp[st_ix_beta + k]) - log(Beta_rte[k])
				if Z[st_ix_z + k] > maxval:
					maxval = Z[st_ix_z + k]
			for k in range(K):
				Z[st_ix_z + k] = exp(Z[st_ix_z + k] - maxval)
				sumrow += Z[st_ix_z + k]
			for k in range(K):
				Z[st_ix_z + k] *= W[i] / sumrow

	else:
		for i in prange(nW, schedule='static', num_threads=nthreads):
			st_ix_z = i * K
			st_ix_theta = ix_d[i] * K
			st_ix_beta = ix_v[i] * K
			sumrow = 0
			for k in range(K):
				Z[st_ix_z + k] = exp(psi(Theta_shp[st_ix_theta + k]) - log(Theta_rte[k]) + psi(Beta_shp[st_ix_beta + k]) - log(Beta_rte[k]))
				sumrow += Z[st_ix_z + k]
			for k in range(K):
				Z[st_ix_z + k] *= W[i] / sumrow


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void update_Z_const_pred(float* Z, float* Theta_rte, float* Beta_shp, float* Beta_rte,
							  size_t* ix_d, size_t* ix_v, size_t nW, size_t K, int nthreads) nogil:
	cdef size_t i, k
	cdef size_t st_ix_z, st_ix_beta

	for i in prange(nW, schedule='static', num_threads=nthreads):
		st_ix_z = i * K
		st_ix_beta = ix_v[i] * K
		for k in range(K):
			Z[st_ix_z + k] = - log(Theta_rte[k]) + psi(Beta_shp[st_ix_beta + k]) - log(Beta_rte[k])

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void update_Z_var_pred(float* Z, float* Zconst, float* W, float* Theta_shp, size_t* ix_d,
							size_t nW, size_t K, int nthreads) nogil:
	cdef size_t i, k
	cdef size_t st_ix_z, st_ix_theta
	cdef float sumrow, maxval

	for i in prange(nW, schedule='static', num_threads=nthreads):
		st_ix_z = i * K
		st_ix_theta = ix_d[i] * K
		sumrow = 0
		maxval = - 10**1
		for k in range(K):
			Z[st_ix_z + k] = Theta_shp[st_ix_theta + k] + Zconst[st_ix_z + k]
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
				   float* Eps_shp, float* Eps_rte, float* R, size_t* ix_u_r, size_t* ix_d_r, int sum_exp_trick,
				   size_t nR, size_t K, int nthreads) nogil:
	cdef size_t i, k
	cdef size_t st_ix_y, st_ix_u, st_ix_d, ix_2
	cdef float E_eta, sumrow, maxval

	if sum_exp_trick:
		for i in prange(nR, schedule='static', num_threads=nthreads):
			st_ix_u = ix_u_r[i] * K
			st_ix_d = ix_d_r[i] * K
			st_ix_y = i * K
			sumrow = 0
			maxval = - 10**11
			for k in range(K):
				E_eta = psi(Eta_shp[st_ix_u + k]) - log(Eta_rte[k])
				ix_2 = st_ix_d + k
				Ya[st_ix_y + k] = E_eta + psi(Theta_shp[ix_2]) - log(Theta_rte[k])
				Yb[st_ix_y + k] = E_eta + psi(Eps_shp[ix_2]) - log(Eps_rte[k])
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

	else:

		for i in prange(nR, schedule='static', num_threads=nthreads):
			st_ix_u = ix_u_r[i] * K
			st_ix_d = ix_d_r[i] * K
			st_ix_y = i * K
			sumrow = 0
			for k in range(K):
				E_eta = psi(Eta_shp[st_ix_u + k]) - log(Eta_rte[k])
				ix_2 = st_ix_d + k
				Ya[st_ix_y + k] = exp(E_eta + psi(Theta_shp[ix_2]) - log(Theta_rte[k]))
				Yb[st_ix_y + k] = exp(E_eta + psi(Eps_shp[ix_2]) - log(Eps_rte[k]))
				sumrow += Ya[st_ix_y + k] + Yb[st_ix_y + k]
			for k in range(K):
				Ya[st_ix_y + k] *= R[i] / sumrow
				Yb[st_ix_y + k] *= R[i] / sumrow


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void update_Y_csr(float* Ya, float* Yb, float* Eta_shp, float* Eta_rte,
					   float* Theta_shp, float* Theta_rte, float* Eps_shp, float* Eps_rte,
					   float* R, size_t* ix_u_r, size_t* docs_batch, size_t* st_ix_doc,
					   size_t ndocs, size_t K, int nthreads) nogil:
	cdef size_t d, did, nobs, i, k
	cdef size_t st_ix_y, st_ix_u, st_ix_d, ix_2
	cdef float E_eta, sumrow, maxval

	## comment: using schedule='dynamic' results in NA values
	for d in prange(ndocs, schedule='static', num_threads=nthreads):
		did = docs_batch[d]
		nobs = st_ix_doc[did + 1] - st_ix_doc[did]
		st_ix_d = did * K
		for i in range(nobs):
			st_ix_u = ix_u_r[i + st_ix_doc[did]] * K
			st_ix_y = i * K
			sumrow = 0
			maxval = - 10**11
			for k in range(K):
				E_eta = psi(Eta_shp[st_ix_u + k]) - log(Eta_rte[k])
				ix_2 = st_ix_d + k
				Ya[st_ix_y + k] = E_eta + psi(Theta_shp[ix_2]) - log(Theta_rte[k])
				Yb[st_ix_y + k] = E_eta + psi(Eps_shp[ix_2]) - log(Eps_rte[k])
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
						   size_t* ix_d_w, size_t* ix_d_r, size_t nW, size_t nR, size_t K,
						   int allow_inconsistent, int nthreads) nogil:
	cdef size_t i, j, k
	cdef size_t st_ix_theta, st_ix_z, st_ix_y
	if allow_inconsistent:
		for i in prange(nW, schedule='static', num_threads=nthreads):
			st_ix_theta = ix_d_w[i] * K
			st_ix_z = i * K
			for k in range(K):
				Theta_shp[st_ix_theta + k] += Z[st_ix_z + k]
		for j in prange(nR, schedule='static', num_threads=nthreads):
			st_ix_theta = ix_d_r[j] * K
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
			st_ix_theta = ix_d_r[j] * K
			st_ix_y = j * K
			for k in range(K):
				Theta_shp[st_ix_theta + k] += Ya[st_ix_y + k]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void update_Theta_shp_wuser(float* Theta_shp, float* Z, float* Ya, float* Yb,
						   size_t* ix_d_w, size_t* ix_d_r, size_t nW, size_t nR, size_t K,
						   int allow_inconsistent, int nthreads) nogil:
	cdef size_t i, j, k
	cdef size_t st_ix_theta, st_ix_z, st_ix_y
	if allow_inconsistent:
		for i in prange(nW, schedule='static', num_threads=nthreads):
			st_ix_theta = ix_d_w[i] * K
			st_ix_z = i * K
			for k in range(K):
				Theta_shp[st_ix_theta + k] += Z[st_ix_z + k]
		for j in prange(nR, schedule='static', num_threads=nthreads):
			st_ix_theta = ix_d_r[j] * K
			st_ix_y = j * K
			for k in range(K):
				Theta_shp[st_ix_theta + k] += Ya[st_ix_y + k] + Yb[st_ix_y + k]
	else:
		for i in range(nW):
			st_ix_theta = ix_d_w[i] * K
			st_ix_z = i * K
			for k in range(K):
				Theta_shp[st_ix_theta + k] += Z[st_ix_z + k]
		for j in range(nR):
			st_ix_theta = ix_d_r[j] * K
			st_ix_y = j * K
			for k in range(K):
				Theta_shp[st_ix_theta + k] += Ya[st_ix_y + k] + Yb[st_ix_y + k]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void update_Theta_shp_pred(float* Theta_shp, float* Z, size_t* ix_d,
								size_t nW, size_t K, int nthreads) nogil:
	cdef size_t i, k
	cdef size_t st_ix_theta, st_ix_z
	for i in prange(nW, schedule='static', num_threads=nthreads):
		st_ix_theta = ix_d[i] * K
		st_ix_z = i * K
		for k in range(K):
			Theta_shp[st_ix_theta + k] += Z[st_ix_z + k]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void update_Beta_shp(float* Beta_shp, float* Z, size_t* ix_v, size_t nW, size_t K, int nthreads) nogil:
	cdef size_t i, k
	cdef size_t st_ix_beta, st_ix_Z
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
						 size_t* ix_u, size_t nR, size_t K, int nthreads) nogil:
	cdef size_t i, k
	cdef size_t st_ix_y, st_ix_eta
	for i in prange(nR, schedule='static', num_threads=nthreads):
		st_ix_eta = ix_u[i] * K
		st_ix_y = i * K
		for k in range(K):
			Eta_shp[st_ix_eta + k] += Ya[st_ix_y + k] + Yb[st_ix_y + k]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void update_Eps_shp(float* Eps_shp, float* Yb, size_t* ix_d, size_t nR, size_t K,
						 int allow_inconsistent, int nthreads) nogil:
	cdef size_t i, k
	cdef size_t st_ix_eps, st_ix_y
	if allow_inconsistent:
		for i in prange(nR, schedule='static', num_threads=nthreads):
			st_ix_eps = ix_d[i] * K
			st_ix_y = i * K
			for k in range(K):
				Eps_shp[st_ix_eps + k] += Yb[st_ix_y + k]
	else:
		for i in range(nR):
			st_ix_eps = ix_d[i] * K
			st_ix_y = i * K
			for k in range(K):
				Eps_shp[st_ix_eps + k] += Yb[st_ix_y + k]

## this function was copy-pasted from hpfrec, thus the variable names
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void llk_plus_rmse(float* T, float* B, float* Y,
						size_t* ix_u, size_t* ix_i, size_t nY, size_t kszt,
						long double* out, int nthreads, int add_mse, int full_llk) nogil:
	cdef size_t i
	cdef int one = 1
	cdef float yhat
	cdef long double out1 = 0
	cdef long double out2 =  0
	cdef int k = <int> kszt
	if add_mse:
		if full_llk:
			for i in prange(nY, schedule='static', num_threads=nthreads):
				yhat = sdot(&k, &T[ix_u[i] * kszt], &one, &B[ix_i[i] * kszt], &one)
				out1 += Y[i]*log(yhat) - log(gamma(Y[i] + 1))
				out2 += (Y[i] - yhat)**2
		else:
			for i in prange(nY, schedule='static', num_threads=nthreads):
				yhat = sdot(&k, &T[ix_u[i] * kszt], &one, &B[ix_i[i] * kszt], &one)
				out1 += Y[i]*log(yhat)
				out2 += (Y[i] - yhat)**2
		out[0] = out1
		out[1] = out2
	else:
		if full_llk:
			for i in prange(nY, schedule='static', num_threads=nthreads):
				out1 += Y[i]*log(sdot(&k, &T[ix_u[i] * kszt], &one, &B[ix_i[i] * kszt], &one)) - log(gamma(Y[i] + 1))
			out[0] = out1
		else:
			for i in prange(nY, schedule='static', num_threads=nthreads):
				out1 += Y[i]*log(sdot(&k, &T[ix_u[i] * kszt], &one, &B[ix_i[i] * kszt], &one))
			out[0] = out1
	### Comment: adding += directly to *out triggers compiler optimizations that produce
	### different (and wrong) results across different runs.


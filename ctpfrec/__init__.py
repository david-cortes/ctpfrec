import pandas as pd, numpy as np
import cython_loops
import hpfrec

class CTPF:
	def __init__(self, k=30, a=.3, b=.3, c=.3, d=.3,
				 e=.3, f=.3, g=.3, h=.3, nthreads=4):
		self.nthreads = nthreads
		self.k = k
		self.a = a
		self.b = b
		self.c = c
		self.d = d
		self.e = e
		self.f = f
		self.g = g
		self.h = h

	def fit(counts_df, words_df):
		Wdf = words_df.rename(columns={'ItemId':'UserId', 'WordId':'ItemId'})
		initializer = HPF(k = k, keep_all_objs=True)
		initializer.fit(Wdf)
		self.Theta_shp = initializer.Gamma_shp
		self.Theta_rte = initializer.Gamma_rte
		self.Beta_shp = initializer.Lambda_shp
		self.Beta_rte = initializer.Lambda_rte

		self.nusers = counts_df.UserId.max() + 1
		self.nitems = max(counts_df.ItemId.max(), words_df.ItemId.max()) + 1
		self.nwords = words_df.WordId.max() + 1

		self.Eta_shp = self.e + np.random.uniform(-0.1, 0.1, size=(self.nusers, self.k)).astype('float32')
		self.Eta_rte = self.f + np.random.uniform(-0.1, 0.1, size=(self.nusers, self.k)).astype('float32')
		self.Eps_shp = self.g + np.random.uniform(-0.1, 0.1, size=(self.nitems, self.k)).astype('float32')
		self.Eps_rte = self.h + np.random.uniform(-0.1, 0.1, size=(self.nitems, self.k)).astype('float32')
		Z = np.empty((words_df.shape[0], self.k), dtype='float32')
		Ya = np.zeros((df_counts.shape[0], self.k), dtype='float32')
		Yb = np.zeros((df_counts.shape[0], self.k), dtype='float32')

		cy.fit_ctpf(
			self.Theta_shp, self.Theta_rte,
			self.Beta_shp, self.Beta_rte,
			self.Eta_shp, self.Eta_rte,
            self.Eps_shp, self.Eps_rte,
            Z, Ya, Yb,
            counts_df, words_df, self.k,
            self.a, self.b, self.c, self.d,
            self.e, self.f, self.g, self.h,
            self.nthreads
            )

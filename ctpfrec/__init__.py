import pandas as pd, numpy as np
import multiprocessing, os, warnings
import ctpfrec.cy as cy
import ctypes, types, inspect
from hpfrec import HPF, cython_loops
pd.options.mode.chained_assignment = None


class CTPF:
	"""
	Collaborative Topic Poisson Factorization

	Model for recommending items based on probabilistic Poisson factorization on sparse count
	data (e.g. number of times a user viewed different items) along with count data on item attributes
	(e.g. bag-of-words representation of text descriptions of items), using mean-field variational inference
	with coordinate-ascent. Can also accommodate user attributes in addition to item attributes (see note below
	for more information).

	Can use different stopping criteria for the opimization procedure:

	1) Run for a fixed number of iterations (stop_crit='maxiter').
	2) Calculate the Poisson log-likelihood every N iterations (stop_crit='train-llk' and check_every)
	   and stop once {1 - curr/prev} is below a certain threshold (stop_thr)
	3) Calculate the Poisson log-likelihood in a user-provided validation set (stop_crit='val-llk', val_set, and check_every)
	   and stop once {1 - curr/prev} is below a certain threshold. For this criterion, you might want to lower the
	   default threshold (see Note).
	4) Check the the difference in the Theta matrix after every N iterations (stop_crit='diff-norm', check_every)
	   and stop once the *l2-norm* of this difference is below a certain threshold (stop_thr).
	   Note that this is **not a percent** difference as it is for log-likelihood criteria, so you should put a larger
	   value than the default here.
	   This is a much faster criterion to calculate and is recommended for larger datasets.
	
	If passing reindex=True, it will internally reindex all user and item IDs. Your data will not require
	reindexing if the IDs for users, items, and words (or other countable item attributes) in counts_df and
	words_df meet the following criteria:

	1) Are all integers.
	2) Start at zero.
	3) Don't have any enumeration gaps, i.e. if there is a user '4', user '3' must also be there.

	If you only want to obtain the fitted parameters and use your own API later for recommendations,
	you can pass produce_dicts=False and pass a folder where to save them in csv format (they are also
	available as numpy arrays in this object's Theta, Eta and Epsilon attributes). Otherwise, the model
	will create Python dictionaries with entries for each user, item, and word, which can take quite a bit of
	RAM memory. These can speed up predictions later through this package's API.

	Passing verbose=True will also print RMSE (root mean squared error) at each iteration.
	For slighly better speed pass verbose=False once you know what a good threshold should be
	for your data.

	Note
	----
	DataFrames and arrays passed to '.fit' might be modified inplace - if this is a problem you'll
	need to pass a copy to them, e.g. 'counts_df=counts_df.copy()'.

	Note
	----
	If 'check_every' is not None and stop_crit is not 'diff-norm', it will, every N iterations,
	calculate the Poisson log-likelihood of the data. By default, this is NOT the full likelihood, (not including a
	constant that depends on the data but not on the parameters and which is quite slow to compute). The reason why
	it's calculated by default like this is because otherwise it can result in overflow (number is too big for the data
	type), but be aware that if not adding this constant, the number can turn positive
	and will mess with the stopping criterion for likelihood.

	Note
	----
	If you pass a validation set, it will calculate the Poisson log-likelihood *of the non-zero observations
	only*, rather than the complete Poisson log-likelihood that includes also the combinations of users and items
	not present in the data (assumed to be zero), thus it's more likely that you might see positive numbers here.
	Compared to ALS, iterations from this algorithm are a lot faster to compute, so don't be scared about passing
	large numbers for maxiter.

	Note
	----
	In some unlucky cases, the parameters will become NA in the first iteration, in which case you should see
	weird values for log-likelihood and RMSE. If this happens, try again with a different random seed.

	Note
	----
	As this model will fit the parameters to both user-item interactions and item attributes, you might see the
	Poisson log-likelihood decreasing during iterations. This doesn't necessarily mean that it failed,
	but in such cases you might want to try increasing K or decreasing the number of item attributes.

	Note
	----
	Can also fit a model that includes user attributes in the same format as the bag-of-words representations
	of items - in this case the new variables will be called Omega (user-factor matrix), Kappa (user_attribute-factor),
	X (multinomial for user attributes). The Y variables will be associated as follows:
	Ya (Omega, Theta) - Yb (Eta, Theta) - Yc (Omega, Epsilon) - Yd (Eta, Epsilon).
	The priors for these new parameters will be taken to be the same ones as their item counterparts.

	Parameters
	----------
	k : int
		Number of latent factors (topics) to use.
	a : float
		Shape parameter for the word-topic prior (Beta).
		If fitting the model with user attributes, will also be taken as prior for
		the user_attribute-factor matrix.
	b : float
		Rate parameter for the word-topic prior (Beta).
		If fitting the model with user attributes, will also be taken as prior for
		the user_attribute-factor matrix.
	c : float
		Shape parameter document-topic prior (Theta).
	d : float
		Rate parameter document-topic prior (Theta).
	e : float
		Shape parameter for the user-topic prior (Eta).
	f : float
		Rate parameter for the user-topic prior (Eta).
	g : float
		Shape parameter for the document-topic offset prior (Epsilon).
		If fitting the model with user attributes, will also be taken as prior for
		the user-factor offset matrix.
	h : float
		Rate parameter for the document-topic offset prior (Epsilon).
		If fitting the model with user attributes, will also be taken as prior for
		the user-factor offset matrix.
	stop_crit : str, one of 'maxiter', 'train-llk', 'val-llk', 'diff-norm'
		Stopping criterion for the optimization procedure.
	stop_thr : float
		Threshold for proportion increase in log-likelihood or l2-norm for difference between matrices.
	reindex : bool
		Whether to reindex data internally.
	miniter : int or None
		Minimum number of iterations for which to run the optimization procedure. When using likelihood as a
		stopping criterion, note that as the model is fit to both user-item interactions and item attributes,
		the Poisson likelihood for the interactions alone can decrease during iterations as the complete
		model likelihood increases (and this Poisson likelihood might start increasing again later).
		Thus, a minimum number of iterations will avoid stopping when the Poisson likelihood decreases.
	maxiter : int or None
		Maximum number of iterations for which to run the optimization procedure. This corresponds to epochs when
		fitting in batches of users. Recommended to use a lower number when passing a batch size.
	check_every : None or int
		Calculate log-likelihood every N iterations.
	verbose : bool
		Whether to print convergence messages.
	random_seed : int or None
		Random seed to use when starting the parameters.
	ncores : int
		Number of cores to use to parallelize computations.
		If set to -1, will use the maximum available on the computer.
	initialize_hpf : bool
		Whether to initialize the Theta and Beta matrices using hierarchical Poisson factorization
		on the bag-of-words representation only (words_df passed to 'fit'). This can provide better
		results than a random initialization, but it takes extra time to fit.
	standardize_items : bool
		Whether to standardize the item bag-of-words representations passed to '.fit' ('words_df') so that all the
		items have the same sum of words (set to the mean sum of word counts across items).
		Will also apply to user attributes if passing them to '.fit'.
	rescale_factors : bool
		Whether to rescale the resulting item-factor matrix (Theta) after fitting to have its rows sum to 1.
		This decreases the model susceptibility to have items with more words be considered more popular, but
		it can also result in way worse rankings.
		Will also be applied to user factors if fitting the model with user attributes.
		(Not recommended)
	missing_items : str, one of 'include' or 'exclude'
		If there are items in the 'words_df' object to be passed to '.fit' that are not present in 'counts_df',
		shall they be considered as having all their user-item interactions with a count of zero
		(when passing 'include'), or shall they be considered to be censored (e.g. missing because the model is
		fit to bag-of-words of articles that are not available to users). In the second case, these items will be
		included when initializing with 'initialize_hpf=True', but will be excluded afterwards.
		In the second case, note that the model won't be able to make predictions for these items, but you can
		add them afterwards using the '.add_items' method.
		Same for user attributes when fitting the model with user side information.
		Note that this **only applies to extra items/users with side info but no interaction**, while any
		user-item interaction not present in the data is taken as include.
		Forced to 'include' when passing 'initialize_hpf=False' or 'reindex=False'.
	step_size : None or function -> float in (0, 1)
		Function that takes the iteration/epoch number as input (starting at zero) and produces the step size
		for the update to Beta and Theta. When initializing these through hierarchical Posisson factorization,
		it can be beneficial to have the first steps change them less or not change them at all, while the user
		and offset matrices start getting shaped towards these initialized topics, with later iterations being
		allowed to change them more (so it starts at zero and tends towards 1 as the iteration number increases).
		When using 'stop_crit=diff-norm', it will not stop if step_size(iteration)<=1e-2.
		You can also pass a function that always returns zero if you do not wish to update the Theta and Beta
		parameters obtained from HPF, but in that case you'll also need to change the stopping criterion.
		Will also apply to the Kappa parameter in the model with user attributes.
		Forced to None when passing 'initialize_hpf=False'.
	allow_inconsistent_math : bool
		Whether to allow inconsistent floating-point math (producing slightly different results on each run)
		which would allow parallelization of the updates for all of the shape parameters.
	full_llk : bool
		Whether to calculate the full log-likehood, including terms that don't depend on the model parameters
		(thus are constant for a given dataset).
	keep_data : bool
		Whether to keep information about which user was associated with each item
		in the training set, so as to exclude those items later when making Top-N
		recommendations.
	save_folder : str or None
		Folder where to save all model parameters as csv files.
	produce_dicts : bool
		Whether to produce Python dictionaries for users and items, which
		are used to speed-up the prediction API of this package. You can still predict without
		them, but it might take some additional miliseconds (or more depending on the
		number of users and items).
	sum_exp_trick : bool
		Whether to use the sum-exp trick when scaling the multinomial parameters - that is, calculating them as
		exp(val - maxval)/sum_{val}(exp(val - maxval)) in order to avoid numerical overflow if there are
		too large numbers. For this kind of model, it is unlikely that it will be required, and it adds a
		small overhead, but if you notice NaNs in the results or in the likelihood, you might give this option a try.
		Forced to True when passing 'initialize_hpf=True'.
		Will also be forced to True when passing user side information.
	keep_all_objs : bool
		Whether to keep intermediate objects/variables in the object that are not necessary for
		predictions - these are: Gamma_shp, Gamma_rte, Lambda_shp, Lambda_rte, k_rte, t_rte
		(when passing True here, the model object will have these extra attributes too).
		Without these objects, it's not possible to call functions that alter the model parameters
		given new information after it's already fit.
	
	Attributes
	----------
	Theta : array (nitems, k)
		Item-topic matrix.
	Beta : array (nwords, k)
		Word-topic matrix. Only kept when passing 'keep_all_objs=True'
	Eta : array (nusers, k)
		User-topic matrix
	Epsilon : array (nitems, k)
		Item-topic offset matrix
	user_mapping_ : array (nusers,)
		ID of the user (as passed to .fit) of each row of Eta.
	item_mapping_ : array (nitems,)
		ID of the item (as passed to .fit) of each row of Beta.
	word_mapping_ : array (nwords,)
		ID of the word (as passed to .fit) of each row of Theta and Epsilon.
	user_dict_ : dict (nusers)
		Dictionary with the mapping between user IDs (as passed to .fit) and rows of Eta.
	item_dict_ : dict (nitems)
		Dictionary with the mapping between item IDs (as passed to .fit) and rows of Theta and Epsilon.
	word_dict_ : dict (nwords)
		Dictionary with the mapping between item IDs (as passed to .fit) and rows of Beta.
	is_fitted : bool
		Whether the model has been fit to some data.
	niter : int
		Number of iterations for which the fitting procedure was run.

	References
	----------
	[1] Content-based recommendations with poisson factorization (Gopalan, P.K., Charlin, L. and Blei, D., 2014)
	"""
	
	def __init__(self, k=50, a=.3, b=.3, c=.3, d=.3, e=.3, f=.3, g=.3, h=.3,
				 stop_crit='train-llk', stop_thr=1e-3, reindex=True,
				 miniter=25, maxiter=70, check_every=10, verbose=True,
				 random_seed=None, ncores=-1, initialize_hpf=True,
				 standardize_items=False, rescale_factors=False,
				 missing_items='include', step_size=lambda x: 1-1/np.sqrt(x+1),
				 allow_inconsistent_math=False, full_llk=False,
				 keep_data=True, save_folder=None, produce_dicts=True,
				 sum_exp_trick=False, keep_all_objs=True):

		## checking input
		assert isinstance(k, int)
		if isinstance(a, int):
			a = float(a)
		if isinstance(b, int):
			b = float(b)
		if isinstance(c, int):
			c = float(c)
		if isinstance(d, int):
			d = float(d)
		if isinstance(e, int):
			e = float(e)
		if isinstance(f, int):
			f = float(f)
		if isinstance(g, int):
			g = float(g)
		if isinstance(h, int):
			h = float(h)
			
		assert isinstance(a, float)
		assert isinstance(b, float)
		assert isinstance(c, float)
		assert isinstance(d, float)
		assert isinstance(e, float)
		assert isinstance(f, float)
		assert isinstance(g, float)
		assert isinstance(h, float)
		
		assert k>0
		assert a>0
		assert b>0
		assert c>0
		assert d>0
		assert e>0
		assert f>0
		assert g>0
		assert h>0
		
		if ncores == -1:
			ncores = multiprocessing.cpu_count()
			if ncores is None:
				ncores = 1 
		assert ncores>0
		assert isinstance(ncores, int)

		if random_seed is not None:
			assert isinstance(random_seed, int)

		assert stop_crit in ['maxiter', 'train-llk', 'val-llk', 'diff-norm']
		assert missing_items in ['include', 'exclude']

		if maxiter is not None:
			assert maxiter>0
			assert isinstance(maxiter, int)
		else:
			if stop_crit!='maxiter':
				raise ValueError("If 'stop_crit' is set to 'maxiter', must provide a maximum number of iterations.")
			maxiter = 10**10

		if miniter is not None:
			assert miniter >= 0
			assert isinstance(miniter, int)
		else:
			miniter = 0
			
		if check_every is not None:
			assert isinstance(check_every, int)
			assert check_every>0
			assert check_every<=maxiter
		else:
			if stop_crit != 'maxiter':
				raise ValueError("If 'stop_crit' is not 'maxiter', must input after how many iterations to calculate it.")
			check_every = 0

		if isinstance(stop_thr, int):
			stop_thr = float(stop_thr)
		if stop_thr is not None:
			assert stop_thr>0
			assert isinstance(stop_thr, float)
			
		if save_folder is not None:
			save_folder = os.path.expanduser(save_folder)
			assert os.path.exists(save_folder)
			
		verbose = bool(verbose)
		if (stop_crit == 'maxiter') and (not verbose):
			check_every = 0

		if step_size is not None:
			if not isinstance(step_size, types.FunctionType):
				raise ValueError("'step_size' must be a function.")
			if len(inspect.getfullargspec(step_size).args) < 1:
				raise ValueError("'step_size' must be able to take the iteration number as input.")
			assert (step_size(0) >= 0) and (step_size(0) <= 1)
			assert (step_size(1) >= 0) and (step_size(1) <= 1)
		
		## storing these parameters
		self.k = k
		self.a = a
		self.b = b
		self.c = c
		self.d = d
		self.e = e
		self.f = f
		self.g = g
		self.h = h

		self.ncores = ncores
		self.allow_inconsistent_math = bool(allow_inconsistent_math)
		self.random_seed = random_seed
		self.stop_crit = stop_crit
		self.reindex = bool(reindex)
		self.keep_data = bool(keep_data)
		self.maxiter = maxiter
		self.miniter = miniter
		self.check_every = check_every
		self.stop_thr = stop_thr
		self.save_folder = save_folder
		self.verbose = verbose
		self.produce_dicts = bool(produce_dicts)
		self.full_llk = bool(full_llk)
		self.keep_all_objs = bool(keep_all_objs)
		self.sum_exp_trick = bool(sum_exp_trick)
		self.initialize_hpf = bool(initialize_hpf)
		self.standardize_items = bool(standardize_items)
		self.rescale_factors = bool(rescale_factors)
		self.step_size = step_size
		self.missing_items = missing_items
		if not self.initialize_hpf:
			self.missing_items = 'include'
		else:
			self.sum_exp_trick = True
		if not self.reindex:
			self.missing_items = 'include'

		if not self.reindex:
			self.produce_dicts = False

		if self.standardize_items and self.rescale_factors:
			msg = "You've passed both 'standardize_items=True' and 'rescale_factors=True'."
			msg += " This is a weird combination and you might experience very poor quality predictions"
			warnings.warn(msg)
		
		## initializing other attributes
		self.Theta = None
		self.Beta = None
		self.Eta = None
		self.Epsilon = None
		self.user_mapping_ = None
		self.item_mapping_ = None
		self.word_mapping_ = None
		self.user_dict_ = None
		self.item_dict_ = None
		self.word_dict_ = None
		self.is_fitted = False
		self.niter = None

	def _process_data(self, counts_df, words_df, user_df):
		## TODO: refactor this function, make it mode modular

		self._counts_df = self._check_df(counts_df, ttl='counts_df')
		self._words_df = self._check_df(words_df, ttl='words_df')
		if user_df is not None:
			self._user_df = self._check_df(user_df, ttl='user_df')
			self.sum_exp_trick = True
			
		if self.reindex:
			self._counts_df['UserId'], self.user_mapping_ = pd.factorize(self._counts_df.UserId)
			self._counts_df['ItemId'], self.item_mapping_ = pd.factorize(self._counts_df.ItemId)
			self.nusers = self.user_mapping_.shape[0]
			self.nitems = self.item_mapping_.shape[0]

			self._need_filter_beta = False
			self._need_filter_kappa = False

			words_df_id_orig = self._words_df.ItemId.values.copy()
			cat_ids = pd.Categorical(self._words_df.ItemId, self.item_mapping_)
			ids_new = cat_ids.isnull()
			if ids_new.sum() > 0:
				if (self.missing_items == 'exclude') and (not self.initialize_hpf):
					msg = "'words_df' contains items that are not present in 'counts_df', which will be excluded."
					msg += " If you still wish to include them in the model, use 'missing_items='include''."
					msg += " Any words that were associated with only those items will also be excluded."
					msg += " For information about which words are used by the model, see the attribute 'word_mapping_'."
					warnings.warn(msg)
					self._words_df = self._words_df.loc[~ids_new]
					self._filter_from_words_df = False
					self._words_df['ItemId'] = pd.Categorical(self._words_df.ItemId, self.item_mapping_).codes
				else:
					self._take_ix_item = self.item_mapping_.shape[0]
					new_ids = np.unique(words_df_id_orig[ids_new]).reshape(-1)
					if np.unique(words_df_id_orig).reshape(-1).shape[0] == new_ids.shape[0]:
						raise ValueError("'words_df' contains no items in common with 'counts_df'.")
					self.item_mapping_ = np.r_[self.item_mapping_, new_ids.copy()]
					self._words_df['ItemId'] = pd.Categorical(words_df_id_orig, self.item_mapping_).codes
					self.nitems = self.item_mapping_.shape[0]
					self._filter_from_words_df = True

					if (self.missing_items == 'exclude') and self.initialize_hpf:
						self.item_mapping_ = self.item_mapping_[:self._take_ix_item]
						self.nitems = self.item_mapping_.shape[0]
						words_out = self._words_df.WordId[ids_new].unique()
						words_in = self._words_df.WordId[~ids_new].unique()
						words_exclude = ~np.in1d(words_out, words_in)
						if np.sum(words_exclude) > 0:
							msg = "Some words are associated only with items that are in 'words_df' but not in 'counts_df'."
							msg += " These will be used to initialize Beta but will be excluded from the final model."
							msg += " If you still wish to include them in the model, use 'missing_items='include''."
							msg += " For information about which words are used by the model, see the attribute 'word_mapping_'."
							warnings.warn(msg)
							self._need_filter_beta = True
							self._take_ix_words = words_in.shape[0]
							self.word_mapping_ = np.r_[words_in, words_out[words_exclude]]
							self._words_df['WordId'] = pd.Categorical(self._words_df.WordId.values, self.word_mapping_).codes
							self.word_mapping_ = self.word_mapping_[:self._take_ix_words]


			else:
				self._words_df['ItemId'] = cat_ids.codes.copy()
				self._filter_from_words_df = False
			
			if not self._need_filter_beta:
				self._words_df['WordId'], self.word_mapping_ = pd.factorize(self._words_df.WordId)
			self.nwords = self.word_mapping_.shape[0]

			if user_df is not None:
				user_df_id_orig = self._user_df.UserId.values.copy()
				cat_ids = pd.Categorical(self._user_df.UserId, self.user_mapping_)
				ids_new = cat_ids.isnull()
				if ids_new.sum() > 0:
					if (self.missing_items == 'exclude') and (not self.initialize_hpf):
						msg = "'user_df' contains users that are not present in 'counts_df', which will be excluded."
						msg += " If you still wish to include them in the model, use 'missing_items='include''."
						msg += " Any user attributes that were associated with only those users will also be excluded."
						msg += " For information about which attributes are used by the model, see the attribute 'user_attr_mapping_'."
						warnings.warn(msg)
						self._user_df = self._user_df.loc[~ids_new]
						self._filter_from_user_df = False
						self._user_df['UserId'] = pd.Categorical(self._user_df.UserId, self.user_mapping_).codes
					else:
						self._take_ix_user = self.user_mapping_.shape[0]
						new_ids = np.unique(user_df_id_orig[ids_new]).reshape(-1)
						if np.unique(user_df_id_orig).reshape(-1).shape[0] == new_ids.shape[0]:
							raise ValueError("'user_df' contains no users in common with 'counts_df'.")
						self.user_mapping_ = np.r_[self.user_mapping_, new_ids.copy()]
						self._user_df['UserId'] = pd.Categorical(user_df_id_orig, self.user_mapping_).codes
						self.nusers = self.user_mapping_.shape[0]
						self._filter_from_user_df = True

						if (self.missing_items == 'exclude') and self.initialize_hpf:
							self.user_mapping_ = self.user_mapping_[:self._take_ix_user]
							self.nusers = self.user_mapping_.shape[0]
							attr_out = self._user_df.AttributeId[ids_new].unique()
							attr_in = self._user_df.AttributeId[~ids_new].unique()
							attr_exclude = ~np.in1d(attr_out, attr_in)
							if np.sum(attr_exclude) > 0:
								msg = "Some user attributes are associated only with users that are in 'user_df' but not in 'counts_df'."
								msg += " These will be used to initialize Kappa but will be excluded from the final model."
								msg += " If you still wish to include them in the model, use 'missing_items='include''."
								msg += " For information about which user attributes are used by the model, see the attribute 'user_attr_mapping_'."
								warnings.warn(msg)
								self._need_filter_kappa = True
								self._take_ix_userattr = attr_in.shape[0]
								self.user_attr_mapping_ = np.r_[attr_in, attr_out[attr_exclude]]
								self._user_df['AttributeId'] = pd.Categorical(self._user_df.AttributeId, self.user_attr_mapping_).codes
								self.user_attr_mapping_ = self.user_attr_mapping_[:self._take_ix_userattr]
				else:
					self._user_df['UserId'] = cat_ids.codes.copy()
					self._filter_from_user_df = False
				
				if not self._need_filter_kappa:
					self._user_df['AttributeId'], self.user_attr_mapping_ = pd.factorize(self._user_df.AttributeId)
				self.nuserattr = self.user_attr_mapping_.shape[0]


			self.user_mapping_ = np.array(self.user_mapping_).reshape(-1)
			self.item_mapping_ = np.array(self.item_mapping_).reshape(-1)
			self.word_mapping_ = np.array(self.word_mapping_).reshape(-1)
			if user_df is not None:
				self.user_attr_mapping_ = np.array(self.user_attr_mapping_).reshape(-1)

			if (self.save_folder is not None) and self.reindex:
				if self.verbose:
					print("\nSaving ID mappings...\n")
				pd.Series(self.user_mapping_).to_csv(os.path.join(self.save_folder, 'users.csv'), index=False)
				pd.Series(self.item_mapping_).to_csv(os.path.join(self.save_folder, 'items.csv'), index=False)
				pd.Series(self.word_mapping_).to_csv(os.path.join(self.save_folder, 'words.csv'), index=False)
				if user_df is not None:
					pd.Series(self.user_attr_mapping_).to_csv(os.path.join(self.save_folder, 'user_attr.csv'), index=False)
		
		## when not reindexing
		else:
			if user_df is None:
				self.nusers = self._counts_df.UserId.max() + 1
			else:
				self.nusers = max(self._counts_df.UserId.max(), self._user_df.UserId.max()) + 1
				self.nuserattr = self._user_df.AttributeId.max() + 1
			self.nitems = max(self._counts_df.ItemId.max(), self._words_df.ItemId.max()) + 1
			self.nwords = self._words_df.WordId.max() + 1
		
		self._counts_df = self._cast_df(self._counts_df, ttl='counts_df')
		self._words_df = self._cast_df(self._words_df, ttl='words_df')
		if user_df is not None:
			self._user_df = self._cast_df(self._user_df, ttl='user_df')
			self._has_user_df = True
		else:
			self._has_user_df = False
		
		self._save_hyperparams()
		return None

	def _filter_words_df(self):
		if self.reindex:
			if self._filter_from_words_df:
				self._words_df = self._words_df.loc[self._words_df.ItemId < self.nitems]
				self._words_df.reset_index(drop=True, inplace=True)
				
				## this is a double chek but should be done elsewhere
				if (self.item_mapping_.shape[0] > self._take_ix_item) or (self.Theta_shp.shape[0] > self._take_ix_item):
					self.item_mapping_ = self.item_mapping_[:self._take_ix_item]

					self.Theta = self.Theta[:self._take_ix_item, :]
					self.Theta_shp = self.Theta_shp[:self._take_ix_item, :]
					self.Theta_rte = self.Theta_rte[:self._take_ix_item, :]
					self.Epsilon = self.Epsilon[:self._take_ix_item, :]
					self.Epsilon_shp = self.Epsilon_shp[:self._take_ix_item, :]
					self.Epsilon_rte = self.Epsilon_rte[:self._take_ix_item, :]

	def _filter_user_df(self):
		if self.reindex and self._has_user_df:
			if self._filter_from_user_df:
				self._user_df = self._user_df.loc[self._user_df.UserId < self.nusers]
				self._user_df.reset_index(drop=True, inplace=True)
				
				## this is a double chek but should be done elsewhere
				if (self.user_mapping_.shape[0] > self._take_ix_user) or (self.Eta_shp.shape[0] > self._take_ix_user):
					self.user_mapping_ = self.user_mapping_[:self._take_ix_user]

					self.Eta = self.Eta[:self._take_ix_user, :]
					self.Eta_shp = self.Eta_shp[:self._take_ix_user, :]
					self.Eta_rte = self.Eta_rte[:self._take_ix_user, :]
					self.Omega = self.Omega[:self._take_ix_user, :]
					self.Omega_shp = self.Omega_shp[:self._take_ix_user, :]
					self.Omega_rte = self.Omega_rte[:self._take_ix_user, :]

	def _save_hyperparams(self):
		if self.save_folder is not None:
			with open(os.path.join(self.save_folder, "hyperparameters.txt"), "w") as pf:
				pf.write("a: %.3f\n" % self.a)
				pf.write("b: %.3f\n" % self.b)
				pf.write("c: %.3f\n" % self.c)
				pf.write("d: %.3f\n" % self.d)
				pf.write("e: %.3f\n" % self.e)
				pf.write("f: %.3f\n" % self.f)
				pf.write("g: %.3f\n" % self.g)
				pf.write("h: %.3f\n" % self.h)
				
				pf.write("k: %d\n" % self.k)
				if self.random_seed is not None:
					pf.write("random seed: %d\n" % self.random_seed)
				else:
					pf.write("random seed: None\n")

	def _unexpected_err_msg(self):
		msg = "Something went wrong. Please open an issue in GitHub indicating the function you called and the constructor parameters."
		raise ValueError(msg)

	def _filter_zero_obs(self, df, ttl='words_df', subj='item or word'):
		if self.stop_crit in ['maxiter', 'diff-norm']:
			thr = 0
		else:
			thr = 0.9
		obs_zero = df.Count.values <= thr
		if obs_zero.sum() > 0:
			msg = "'" + ttl + "' contains observations with a count value less than one, these will be ignored."
			msg += " Any " + subj + " associated exclusively with zero-value observations will be excluded."
			msg += " If using 'reindex=False', make sure that your data still meets the necessary criteria."
			warnings.warn(msg)
			df = df.loc[~obs_zero]
		return df

	def _standardize_counts(self, df, col1='ItemId', col2='WordId'):
		if self.standardize_items:
			sum_by_item = df.groupby(col1)[col2].sum()
			if self.is_fitted:
				if (col1=='UserId') and (col2=='ItemId'):
					target_factor = self.rescale_const_counts_df_
				elif (col1=='ItemId') and (col2=='WordId'):
					target_factor = self.rescale_const_words_df_
				elif (col1=='UserId') and (col2=='AttributeId'):
					target_factor = self.rescale_const_user_df_
				else:
					self._unexpected_err_msg()
			else:
				target_factor = sum_by_item.mean()
				if (col1=='UserId') and (col2=='ItemId'):
					self.rescale_const_counts_df_ = target_factor
				elif (col1=='ItemId') and (col2=='WordId'):
					self.rescale_const_words_df_ = target_factor
				elif (col1=='UserId') and (col2=='AttributeId'):
					self.rescale_const_user_df_ = target_factor
				else:
					self._unexpected_err_msg()
			df = pd.merge(df, sum_by_item.to_frame().reset_index(drop=False).rename(columns={col2:'SumCounts'}))
			df['Count'] = target_factor * df.Count.values / df.SumCounts.values
			df = df[[col1, col2, 'Count']]
		return df
		# take_obs = df.Count.values >= 0.85
		# nitems_before = df[col1].unique().shape[0]
		# nwords_before = df[col2].unique().shape[0]
		# df = df.loc[~take_obs]
		# nitems_after = df[col1].unique().shape[0]
		# nwords_after = df[col2].unique().shape[0]
		# if (nitems_before - nitems_after) > 0:
		# 	warnings.warn(str(nitems_before - nitems_after) + " items discarded after filtering counts on standardized attributes.")
		# if (nwords_before - nwords_after) > 0:
		# 	warnings.warn(str(nwords_before - nwords_after) + " words discarded after filtering counts on standardized attributes.")

	def _cols_from_ttl(self, ttl):
		if ttl == 'counts_df':
			subj = 'user or item'
			col1 = 'UserId'
			col2 = 'ItemId'
		elif ttl == 'words_df':
			subj = 'item or word'
			col1 = 'ItemId'
			col2 = 'WordId'
		elif ttl == 'user_df':
			subj = 'user or user-attribute'
			col1 = 'UserId'
			col2 = 'AttributeId'
		else:
			self._unexpected_err_msg()

		return col1, col2, subj

	def _cast_df(self, df, ttl):
		col1, col2, subj = self._cols_from_ttl(ttl)
		df[col1] = df[col1].values.astype(ctypes.c_int)
		df[col2] = df[col2].values.astype(ctypes.c_int)
		df['Count'] = df.Count.astype('float32')
		return df

	def _check_df(self, df, ttl):
		col1, col2, subj = self._cols_from_ttl(ttl)

		if isinstance(df, np.ndarray):
			assert len(df.shape) > 1
			assert df.shape[1] >= 3
			df = df.values[:,:3]
			df.columns = [col1, col2, "Count"]
			
		if df.__class__.__name__ == 'DataFrame':
			assert df.shape[0] > 0
			assert col1 in df.columns.values
			assert col2 in df.columns.values
			assert 'Count' in df.columns.values
			df = df[[col1, col2, 'Count']]
		else:
			raise ValueError("'" + ttl + "' must be a pandas data frame or a numpy array")

		if self.reindex:
			df = self._filter_zero_obs(df, ttl=ttl, subj=subj)
		if self.standardize_items and (ttl != 'counts_df'):
			df = self._standardize_counts(df, col1=col1, col2=col2)

		return df

	def _process_extra_df(self, df, ttl, df2=None):
		assert self.is_fitted
		assert self.keep_all_objs
		df = self._check_df(df, ttl=ttl)
		nobs_before = df.shape[0]
		col1, col2, subj = self._cols_from_ttl(ttl)
		subj1, temp, subj2 = subj.split()
		del temp
		if self.reindex:
			if ttl == 'counts_df':
				curr_mapping1 = self.user_mapping_
				curr_mapping2 = self.item_mapping_
			elif ttl == 'words_df':
				curr_mapping1 = self.item_mapping_
				curr_mapping2 = self.word_mapping_
			elif ttl == 'user_df':
				curr_mapping1 = self.user_mapping_
				curr_mapping2 = self.user_attr_mapping_
			else:
				self._unexpected_err_msg()

			df[col2] = pd.Categorical(df[col2].values, curr_mapping2).codes
			new_ids2 = df[col2].values == -1
			if new_ids2.sum() > 0:
				df = df.loc[~new_ids2].reset_index(drop=True)
				if df.shape[0] > 0:
					msg = "'" + ttl + "' has " + subj2 + "s that were not present in the training data."
					msg += " These will be ignored."
					warnings.warn(msg)
				else:
					raise ValueError("'" + ttl + "' must contain " + subj2 + "s from the training set.")

			new_ids1 = df[col1].unique()
			repeated = np.in1d(new_ids1, curr_mapping1)
			if repeated.sum() > 0:
				repeated = new_ids1[repeated]
				df = df.loc[~np.in1d(df[col1].values, repeated)].reset_index(drop=True)
				if df.shape[0] > 0:
					msg = "'" + ttl + "' contains " + subj1 + "s that were already present in the training set."
					msg += " These will be ignored."
					warnings.warn(msg)
				else:
					raise ValueError("'" + ttl + "' doesn't contain any new " + subj1 + "s.")

			## this covers the case of passing both user_df and counts_df
			if df2 is not None:
				ttl2 = "counts_df"
				df2 = self._check_df(df2, ttl=ttl2)

				df2['ItemId'] = pd.Categorical(df2[col1].values, self.item_mapping_).codes
				invalid_items = df2.ItemId == -1
				if invalid_items.sum() > 0:
					df2 = df2.loc[~invalid_items].reset_index(drop=True)
					if df2.shape[0] > 0:
						msg = "'" + ttl2 + "' has " + "item" + "s that were not present in the training data."
						msg += " These will be ignored."
						warnings.warn(msg)
					else:
						raise ValueError("'" + ttl2 + "' must contain " + "items" + "s from the training set.")

				new_ids11 = df2[col1].unique()
				repeated = np.in1d(new_ids11, curr_mapping1)
				if repeated.sum() > 0:
					repeated = new_ids1[repeated]
					df2 = df2.loc[~np.in1d(df2[col1].values, repeated)].reset_index(drop=True)
					if df2.shape[0] > 0:
						msg = "'" + ttl2 + "' contains " + subj1 + "s that were already present in the training set."
						msg += " These will be ignored."
						warnings.warn(msg)
					else:
						raise ValueError("'" + ttl2 + "' doesn't contain any new " + subj1 + "s.")

				new_ids1 = np.unique(np.r_[new_ids1, new_ids11])
				new_mapping = np.r_[curr_mapping1, new_ids1]
				df[col1] = pd.Categorical(df[col1].values, new_mapping).codes
				df2[col1] = pd.Categorical(df2[col1].values, new_mapping).codes
				df2 = self._cast_df(df2, ttl=ttl2)
			else:
				new_mapping = np.r_[curr_mapping1, new_ids1]
				df[col1] = pd.Categorical(df[col1].values, new_mapping).codes
		
		else:
			new_item_mapping = None

		df = self._cast_df(df, ttl=ttl)
		if self.standardize_items and (ttl != 'counts_df') and (nobs_before > df.shape[0]):
			df = self._standardize_counts(df, col1=col1, col2=col2)

		if df2 is None:
			return df, new_mapping
		else:
			return df, df2, new_mapping

	def _store_metadata(self):
		self.seen = self._counts_df[['UserId', 'ItemId']].copy()
		self.seen.sort_values(['UserId', 'ItemId'], inplace=True)
		self.seen.reset_index(drop = True, inplace = True)
		self._n_seen_by_user = self.seen.groupby('UserId')['ItemId'].agg(lambda x: len(x)).values
		self._st_ix_user = np.cumsum(self._n_seen_by_user)
		self._st_ix_user = np.r_[[0], self._st_ix_user[:self._st_ix_user.shape[0]-1]]
		self.seen = self.seen.ItemId.values
		return None

	def _exclude_missing_from_index(self):
		## this is a double check but should be done elsewhere
		if self.reindex:
			if self._need_filter_beta:
				if self.Beta_shp.shape[0] > self._take_ix_words:
					self.Beta_shp = self.Beta_shp[:self._take_ix_words, :]
				del self._take_ix_words
			del self._need_filter_beta

			if self._has_user_df:
				if self._need_filter_kappa:
					if self.Kappa_shp.shape[0] > self._take_ix_userattr:
						self.Kappa_shp = self.Kappa_shp[:self._take_ix_userattr, :]
					del self._take_ix_userattr
				del self._need_filter_kappa
		return None

	def _initalize_parameters(self):
		## TODO: make this function more modular

		if self.random_seed is not None:
			np.random.seed(self.random_seed)

		if self.initialize_hpf:
			if self.verbose:
				print("Initializing Theta and Beta through HPF...")
				print("")
			h = HPF(k=self.k, verbose=self.verbose, reindex=True, produce_dicts=False, stop_crit='diff-norm',
					stop_thr=self.stop_thr, random_seed=self.random_seed, keep_all_objs=False,
					sum_exp_trick=self.sum_exp_trick, allow_inconsistent_math=self.allow_inconsistent_math,)
			h.fit(self._words_df.rename(columns={'ItemId':'UserId', 'WordId':'ItemId'}).copy())
			if (h.nusers == self.nitems) and (h.nitems == self.nwords):
				## if using missing_items='include', it should always enter this section
				order_theta = np.argsort(h.user_mapping_)
				order_beta = np.argsort(h.item_mapping_)
				self.Theta_shp = h.Theta[order_theta].copy()
				self.Beta_shp = h.Beta[order_beta].copy()
				del h
				self.Theta_rte = self.d + self.Beta_shp.sum(axis=0, keepdims=True)
				self.Beta_rte = self.b + self.Theta_shp.sum(axis=0, keepdims=True)
			else:
				if self.reindex:
					## from self._process_data, all items that are in words_df but not in counts_df should have
					## numeration greater than the last (maximum) ID in counts_df
					items_take = h.user_mapping_ < self.nitems
					words_take = h.item_mapping_ < self.nwords
				else:
					ids_counts_df = self._counts_df.ItemId.unique()
					items_take = np.in1d(h.user_mapping_, ids_counts_df)
					## for which words to take, need to forcibly determine intersection
					items_words_df = self._words_df.ItemId.unique()
					items_intersect = np.in1d(items_words_df, items_counts_df)
					words_include = self._words_df.WordId.loc[np.in1d(self._words_df.ItemId, items_words_df[items_intersect])].unique()
					words_take = pd.Categorical(words_include, h.item_mapping_).codes

				self.Theta_shp = (self.c * 2*np.random.beta(20, 20, size=(self.nitems, self.k))).astype('float32')
				self.Theta_shp[h.user_mapping_[items_take],:] = h.Theta[items_take].copy()

				self.Beta_shp = (self.a * 2*np.random.beta(20, 20, size=(self.nwords, self.k))).astype('float32')
				self.Beta_shp[h.item_mapping_[words_take],:] = h.Beta[words_take].copy()
				
				self.Theta_rte = self.d + self.Beta_shp.sum(axis=0, keepdims=True)
				self.Beta_rte = self.b + self.Theta_shp.sum(axis=0, keepdims=True)

			if np.isnan(self.Theta_shp).sum().sum() > 0:
				warnings.warn("NaNs produced in initialization of Theta, will use a random start.")
				self.Theta_shp = (self.c * 2*np.random.beta(20, 20, size=(self.nitems, self.k))).astype('float32')
				self.Theta_rte = (self.d * 2*np.random.beta(20, 20, size=(1, self.k))).astype('float32')
			if np.isnan(self.Beta_shp).sum().sum() > 0:
				warnings.warn("NaNs produced in initialization of Beta, will use a random start.")
				self.Beta_shp = (self.a * 2*np.random.beta(20, 20, size=(self.nwords, self.k))).astype('float32')
				self.Beta_rte = (self.b * 2*np.random.beta(20, 20, size=(1, self.k))).astype('float32')
			if self.verbose:
				print("**********************************")
				print("")

			if self._has_user_df:
				self.Omega_shp = (self.e * 2*np.random.beta(20, 20, size=(self.nusers, self.k))).astype('float32')
				self.Omega_rte = (self.f * 2*np.random.beta(20, 20, size=(1, self.k))).astype('float32')
				if self.verbose:
					print("Initializing Kappa through HPF...")
					print("")
				h = HPF(k=self.k, verbose=self.verbose, reindex=True, produce_dicts=False, stop_crit='diff-norm',
					stop_thr=self.stop_thr, random_seed=self.random_seed, keep_all_objs=False,
					sum_exp_trick=self.sum_exp_trick, allow_inconsistent_math=self.allow_inconsistent_math)
				h.fit(self._user_df.rename(columns={'AttributeId':'ItemId'}).copy())
				
				if h.nitems == self.nuserattr:
					## if using missing_items='include', it should always enter this section
					order_kappa = np.argsort(h.item_mapping_)
					self.Kappa_shp = h.Beta[order_kappa].copy()
					self.Kappa_rte = self.b + h.Theta.sum(axis=0, keepdims=True)
					del h
				else:
					if self.reindex:
						attr_take = h.user_attr_mapping_ < self.nuserattr
					else:
						users_counts_df = self._counts_df.UserId.unique()
						users_user_df = self._user_df.UserId.unique()
						users_intersect = np.in1d(users_user_df, users_counts_df)
						attr_include = self._user_df.AttributeId.loc[np.in1d(self._user_df.UserId, users_user_df[users_intersect])].unique()
						attr_take = pd.Categorical(attr_include, h.item_mapping_).codes

					self.Kappa_shp = (self.a * 2*np.random.beta(20, 20, size=(self.nuserattr, self.k))).astype('float32')
					self.Kappa_shp[h.item_mapping_[attr_take],:] = h.Beta[attr_take].copy()
					self.Kappa_rte = self.b + h.Theta.sum(axis=0, keepdims=True)
					del h


				if np.isnan(self.Kappa_shp).sum().sum() > 0:
					warnings.warn("NaNs produced in initialization of Kappa, will use a random start.")
					self.Kappa_shp = (self.a * 2*np.random.beta(20, 20, size=(self.nuserattr, self.k))).astype('float32')
					self.Kappa_rte = (self.b * 2*np.random.beta(20, 20, size=(1, self.k))).astype('float32')
				if self.verbose:
					print("**********************************")
					print("")
			else:
				self.Kappa_shp = np.empty((0,0), dtype='float32')
				self.Kappa_rte = np.empty((0,0), dtype='float32')
		else:
			self.Beta_shp = (self.a * 2*np.random.beta(20, 20, size=(self.nwords, self.k))).astype('float32')
			self.Theta_shp = (self.c * 2*np.random.beta(20, 20, size=(self.nitems, self.k))).astype('float32')
			self.Beta_rte = (self.b * 2*np.random.beta(20, 20, size=(1, self.k))).astype('float32')
			self.Theta_rte = (self.d * 2*np.random.beta(20, 203, size=(1, self.k))).astype('float32')
			if self._has_user_df:
				self.Kappa_shp = (self.a * 2*np.random.beta(20, 20, size=(self.nuserattr, self.k))).astype('float32')
				self.Kappa_rte = (self.b * 2*np.random.beta(20, 20, size=(1, self.k))).astype('float32')
			else:
				self.Kappa_shp = np.empty((0,0), dtype='float32')
				self.Kappa_rte = np.empty((0,0), dtype='float32')
		
		self.Eta_shp = (self.e * 2*np.random.beta(20, 20, size=(self.nusers, self.k))).astype('float32')
		self.Epsilon_shp = (self.g * 2*np.random.beta(20, 20, size=(self.nitems, self.k))).astype('float32')
		self.Eta_rte = (self.f * 2*np.random.beta(20, 20, size=(1, self.k))).astype('float32')
		self.Epsilon_rte = (self.h * 2*np.random.beta(20, 20, size=(1, self.k))).astype('float32')
		if self._has_user_df:
			self.Omega_shp = (self.e * 2*np.random.beta(20, 20, size=(self.nusers, self.k))).astype('float32')
			self.Omega_rte = (self.f * 2*np.random.beta(20, 20, size=(1, self.k))).astype('float32')
		else:
			self.Omega_shp = np.empty((0,0), dtype='float32')
			self.Omega_rte = np.empty((0,0), dtype='float32')

		self._divide_parameters(add_beta=False)

	def _divide_parameters(self, add_beta=False):
		self.Theta = self.Theta_shp / self.Theta_rte
		self.Eta = self.Eta_shp / self.Eta_rte
		self.Epsilon = self.Epsilon_shp / self.Epsilon_rte
		self.Omega = self.Omega_shp / self.Omega_rte
		if add_beta:
			self.Beta = self.Beta_shp / self.Beta_rte
			self.Kappa = self.Kappa_shp / self.Kappa_rte

	def _rescale_parameters(self):
		self.Theta_shp /= self.Theta_shp.sum(axis=1, keepdims=True)
		if self._has_user_df:
			self.Omega_shp /= self.Omega_shp.sum(axis=1, keepdims=True)

	def _clear_internal_objects(self):
		del self._counts_df, self._words_df, self._user_df
		del self.val_set
		if not self._has_user_df:
			del self.Kappa_shp, self.Kappa_rte
			del self.Omega_shp, self.Omega_rte
		if not self.keep_all_objs:
			del self.Theta_shp, self.Theta_rte
			del self.Beta_shp, self.Beta_rte
			del self.Eta_shp, self.Eta_rte
			del self.Epsilon_shp, self.Epsilon_rte

		if self._has_user_df:
			self._M1 = self.Eta + self.Omega
		else:
			self._M1 = self.Eta
		self._M2 = self.Theta + self.Epsilon
		if not self.keep_all_objs:
			del self.Theta, self.Epsilon, self.Omega

		if self.reindex:
			if self._filter_from_words_df:
				del self._take_ix_item
			del self._filter_from_words_df
			if self._has_user_df:
				if self._filter_from_user_df:
					del self._take_ix_user
				del self._filter_from_user_df

	def fit(self, counts_df, words_df, user_df=None, val_set=None):
		"""
		Fit Collaborative Topic Poisson Factorization model to sparse count data

		Note
		----
		DataFrames and arrays passed to '.fit' might be modified inplace - if this is a problem you'll
		need to pass a copy to them, e.g. 'counts_df=counts_df.copy()'.

		Note
		----
		Forcibly terminating the procedure should still keep the last calculated shape and rate
		parameter values, but is not recommended. If you need to make predictions on a forced-terminated
		object, set the attribute 'is_fitted' to 'True'.

		Parameters
		----------
		counts_df : DatFrame or array (n_samples, 3)
			User-Item interaction data with one row per non-zero observation, consisting of triplets ('UserId', 'ItemId', 'Count').
			Must containin columns 'UserId', 'ItemId', and 'Count'.
			Combinations of users and items not present are implicitly assumed to be zero by the model.
		words_df : DatFrame or array (n_samples, 3)
			Bag-of-word representation of items with one row per present unique word, consisting of triplets ('ItemId', 'WordId', 'Count').
			Must contain columns 'ItemId', 'WordId', and 'Count'.
			Combinations of items and words not present are implicitly assumed to be zero.
		val_set : DatFrame or array (n_samples, 3)
			Validation set on which to monitor log-likelihood. Same format as counts_df.
		user_df : DatFrame or array (n_samples, 3)
			User attributes, same format as 'words_df'. Must contain columns 'UserId', 'AttributeId', 'Count'.

		Returns
		-------
		self : obj
			Copy of this object
		"""
		## a basic check
		if self.stop_crit == 'val-llk':
			if val_set is None:
				raise ValueError("If 'stop_crit' is set to 'val-llk', must provide a validation set.")

		## running each sub-process
		if self.verbose:
			self._print_st_msg()
		self._process_data(counts_df, words_df, user_df)
		if self.verbose:
			self._print_data_info()
		if (val_set is not None) and (self.stop_crit!='diff-norm') and (self.stop_crit!='train-llk'):
			HPF._process_valset(self, val_set)
		else:
			self.val_set = pd.DataFrame({
				'UserId': np.empty(0, dtype=ctypes.c_int),
				'ItemId': np.empty(0, dtype=ctypes.c_int),
				'Count': np.empty(0, dtype='float32')})
		if not self._has_user_df:
			self._user_df = pd.DataFrame({'UserId':np.empty(0, dtype=ctypes.c_int),
				'AttributeId':np.empty(0, dtype=ctypes.c_int),
				'Count':np.empty(0, dtype='float32')})
		if self.verbose:
			print("Initializing parameters...")
		self._initalize_parameters()
		self._divide_parameters(add_beta=False)
		if self.missing_items == 'exclude':
			self._exclude_missing_from_index()
			self._filter_words_df()
			if self._has_user_df:
				self._filter_user_df()
			else:
				self._user_df = pd.DataFrame({'UserId':np.empty(0, dtype=ctypes.c_int),
											  'AttributeId':np.empty(0, dtype=ctypes.c_int),
											  'Count':np.empty(0, dtype='float32')})

		## fitting the model
		self.niter = cy.fit_ctpf(
			self.Theta_shp, self.Theta_rte, self.Beta_shp, self.Beta_rte,
			self.Eta_shp, self.Eta_rte, self.Epsilon_shp, self.Epsilon_rte,
			self.Omega_shp, self.Omega_rte, self.Kappa_shp, self.Kappa_rte,
			self.Theta, self.Eta, self.Epsilon, self.Omega,
			self._user_df, self._has_user_df,
			self._counts_df, self._words_df, cython_loops.cast_int(self.k), self.step_size,
			cython_loops.cast_int(self.step_size is not None), cython_loops.cast_int(self.sum_exp_trick),
			cython_loops.cast_float(self.a), cython_loops.cast_float(self.b), cython_loops.cast_float(self.c),
			cython_loops.cast_float(self.d), cython_loops.cast_float(self.e), cython_loops.cast_float(self.f),
			cython_loops.cast_float(self.g), cython_loops.cast_float(self.h),
			cython_loops.cast_int(self.ncores), cython_loops.cast_int(self.maxiter),
			cython_loops.cast_int(self.miniter), cython_loops.cast_int(self.check_every),
			self.stop_crit, self.stop_thr, cython_loops.cast_int(self.verbose),
			self.save_folder if self.save_folder is not None else "",
			cython_loops.cast_int(self.allow_inconsistent_math),
			cython_loops.cast_int(self.val_set.shape[0] > 0), cython_loops.cast_int(self.full_llk),
			self.val_set
			)

		## post-processing and clean-up
		if self.rescale_factors:
			self._rescale_parameters()
		self._divide_parameters(self.keep_all_objs)
		self._store_metadata()
		self._clear_internal_objects()
		if self.verbose:
			print("Producing Python dictionaries...")
		if self.produce_dicts and self.reindex:
			self.user_dict_ = {self.user_mapping_[i]:i for i in range(self.user_mapping_.shape[0])}
			self.item_dict_ = {self.item_mapping_[i]:i for i in range(self.item_mapping_.shape[0])}
			self.word_dict_ = {self.word_mapping_[i]:i for i in range(self.word_mapping_.shape[0])}
			if self._has_user_df:
				self.user_attr_dict = {self.user_attr_mapping_[i]:i for i in range(self.user_attr_mapping_.shape[0])}
		self.is_fitted = True

		return self

	def _topN(self, user_vec, n, exclude_seen, items_pool, user=None):
		if items_pool is None:
			allpreds = - (user_vec.dot(self._M2.T))
			if exclude_seen:
				n_ext = np.min([n + self._n_seen_by_user[user], self._M2.shape[0]])
				rec = np.argpartition(allpreds, n_ext-1)[:n_ext]
				seen = self.seen[self._st_ix_user[user] : self._st_ix_user[user] + self._n_seen_by_user[user]]
				rec = np.setdiff1d(rec, seen)
				rec = rec[np.argsort(allpreds[rec])[:n]]
				if self.reindex:
					return self.item_mapping_[rec]
				else:
					return rec

			else:
				n = np.min([n, self._M2.shape[0]])
				rec = np.argpartition(allpreds, n-1)[:n]
				rec = rec[np.argsort(allpreds[rec])]
				if self.reindex:
					return self.item_mapping_[rec]
				else:
					return rec

		else:
			if isinstance(items_pool, list) or isinstance(items_pool, tuple):
				items_pool = np.array(items_pool)
			if items_pool.__class__.__name__=='Series':
				items_pool = items_pool.values
			if isinstance(items_pool, np.ndarray):  
				if len(items_pool.shape) > 1:
					items_pool = items_pool.reshape(-1)
				if self.reindex:
					items_pool_reind = pd.Categorical(items_pool, self.item_mapping_).codes
					nan_ix = (items_pool_reind == -1)
					if nan_ix.sum() > 0:
						items_pool_reind = items_pool_reind[~nan_ix]
						msg = "There were " + ("%d" % int(nan_ix.sum())) + " entries from 'item_pool'"
						msg += "that were not in the training data and will be exluded."
						warnings.warn(msg)
					del nan_ix
					if items_pool_reind.shape[0] == 0:
						raise ValueError("No items to recommend.")
					elif items_pool_reind.shape[0] == 1:
						raise ValueError("Only 1 item to recommend.")
					else:
						pass
			else:
				raise ValueError("'items_pool' must be an array.")

			if self.reindex:
				allpreds = - user_vec.dot(self._M2[items_pool_reind].T)
			else:
				allpreds = - user_vec.dot(self._M2[items_pool].T)
			n = np.min([n, items_pool.shape[0]])
			if exclude_seen:
				n_ext = np.min([n + self._n_seen_by_user[user], items_pool.shape[0]])
				rec = np.argpartition(allpreds, n_ext-1)[:n_ext]
				seen = self.seen[self._st_ix_user[user] : self._st_ix_user[user] + self._n_seen_by_user[user]]
				if self.reindex:
					rec = np.setdiff1d(items_pool_reind[rec], seen)
					allpreds = - user_vec.dot(self._M2[rec].T)
					return self.item_mapping_[rec[np.argsort(allpreds)[:n]]]
				else:
					rec = np.setdiff1d(items_pool[rec], seen)
					allpreds = - user_vec.dot(self._M2[rec].T)
					return rec[np.argsort(allpreds)[:n]]
			else:
				rec = np.argpartition(allpreds, n-1)[:n]
				return items_pool[rec[np.argsort(allpreds[rec])]]


	def topN(self, user, n=10, exclude_seen=True, items_pool=None):
		"""
		Recommend Top-N items for a user

		Outputs the Top-N items according to score predicted by the model.
		Can exclude the items for the user that were associated to her in the
		training set, and can also recommend from only a subset of user-provided items.

		Parameters
		----------
		user : obj
			User for which to recommend.
		n : int
			Number of top items to recommend.
		exclude_seen: bool
			Whether to exclude items that were associated to the user in the training set.
		items_pool: None or array
			Items to consider for recommending to the user.
		
		Returns
		-------
		rec : array (n,)
			Top-N recommended items.
		"""
		if isinstance(n, float):
			n = int(n)
		assert isinstance(n ,int)
		if self.reindex:
			if self.produce_dicts:
				try:
					user = self.user_dict_[user]
				except:
					raise ValueError("Can only predict for users who were in the training set.")
			else:
				user = pd.Categorical(np.array([user]), self.user_mapping_).codes[0]
				if user == -1:
					raise ValueError("Can only predict for users who were in the training set.")
		if exclude_seen and not self.keep_data:
			raise Exception("Can only exclude seen items when passing 'keep_data=True' to .fit")

		return self._topN(self._M1[user], n, exclude_seen, items_pool, user)

	def topN_cold(self, user_df, n=10, items_pool=None, maxiter=10, ncores=1, random_seed=1, stop_thr=1e-3):
		"""
		Recommend Top-N items for a user that was not in the training set.

		Note
		----
		This function is only available if fitting a model that uses user attributes.

		Note
		----
		The data passed to this function might be modified inplace. Be sure to pass a copy
		of the 'user_df' object if this is a problem.

		Parameters
		----------
		attributes : data frame (n_samples, 2)
			Attributes of the user. Must have columns 'AttributeId', 'Count'.
		n : int
			Number of top items to recommend.
		items_pool: None or array
			Items to consider for recommending to the user.
		
		Returns
		-------
		rec : array (n,)
			Top-N recommended items.
		"""
		if not self._has_user_df:
			msg = "Can only make recommendations for users without any item interactions"
			msg += " when fitting a model with user attributes."
			raise ValueError(msg)

		assert user_df.__class__.__name__ == "DataFrame"
		user_df['UserId'] = self.nusers

		user_vec, temp = self._predict_user_factors(
								user_df=user_df, maxiter=maxiter, ncores=ncores,
								random_seed=random_seed, stop_thr=stop_thr,
								return_ix=False, return_temp=False
								)
		del temp
		user_vec /= self.Omega_rte

		return self._topN(user_vec.reshape(-1), n, False, items_pool, None)

	def predict(self, user, item):
		"""
		Predict count for combinations of users and items
		
		Note
		----
		You can either pass an individual user and item, or arrays representing
		tuples (UserId, ItemId) with the combinatinons of users and items for which
		to predict (one row per prediction).

		Parameters
		----------
		user : array-like (npred,) or obj
			User(s) for which to predict each item.
		item: array-like (npred,) or obj
			Item(s) for which to predict for each user.
		"""
		assert self.is_fitted
		if isinstance(user, list) or isinstance(user, tuple):
			user = np.array(user)
		if isinstance(item, list) or isinstance(item, tuple):
			item = np.array(item)
		if user.__class__.__name__=='Series':
			user = user.values
		if item.__class__.__name__=='Series':
			item = item.values
			
		if isinstance(user, np.ndarray):
			if len(user.shape) > 1:
				user = user.reshape(-1)
			assert user.shape[0] > 0
			if self.reindex:
				if user.shape[0] > 1:
					user = pd.Categorical(user, self.user_mapping_).codes
				else:
					if self.user_dict_ is not None:
						try:
							user = self.user_dict_[user]
						except:
							user = -1
					else:
						user = pd.Categorical(user, self.user_mapping_).codes[0]
		else:
			if self.reindex:
				if self.user_dict_ is not None:
					try:
						user = self.user_dict_[user]
					except:
						user = -1
				else:
					user = pd.Categorical(np.array([user]), self.user_mapping_).codes[0]
			user = np.array([user])
			
		if isinstance(item, np.ndarray):
			if len(item.shape) > 1:
				item = item.reshape(-1)
			assert item.shape[0] > 0
			if self.reindex:
				if item.shape[0] > 1:
					item = pd.Categorical(item, self.item_mapping_).codes
				else:
					if self.item_dict_ is not None:
						try:
							item = self.item_dict_[item]
						except:
							item = -1
					else:
						item = pd.Categorical(item, self.item_mapping_).codes[0]
		else:
			if self.reindex:
				if self.item_dict_ is not None:
					try:
						item = self.item_dict_[item]
					except:
						item = -1
				else:
					item = pd.Categorical(np.array([item]), self.item_mapping_).codes[0]
			item = np.array([item])

		assert user.shape[0] == item.shape[0]
		
		if user.shape[0] == 1:
			if (user[0] == -1) or (item[0] == -1):
				return np.nan
			else:
				return self._M1[user].dot(self._M2[item].T).reshape(-1)[0]
		else:
			nan_entries = (user == -1) | (item == -1)
			if nan_entries.sum() == 0:
				return (self._M1[user] * self._M2[item]).sum(axis=1)
			else:
				non_na_user = user[~nan_entries]
				non_na_item = item[~nan_entries]
				out = np.empty(user.shape[0], dtype=self._M1.dtype)
				out[~nan_entries] = (self._M1[non_na_user] * self._M2[non_na_item]).sum(axis=1)
				out[nan_entries] = np.nan
				return out

	def predict_item_factors(self, words_df, maxiter=10, ncores=1, random_seed=1, stop_thr=1e-3, return_all=False):
		"""
		Obtain latent factors/topics for items given their bag-of-words representation alone

		Note
		----
		For better results, refit the model again including these items.

		Note
		----
		If passing more than one item, the resulting rows will be in the sorted order of
		the item IDs from user_df (e.g. if users are 'b', 'a', 'c', the first row will contain
		the factors for item 'a', second for 'b', third for 'c'.

		Note
		----
		This function is prone to producing all NaNs values.

		Parameters
		----------
		words_df : DataFrame (n_samples, 3)
			Bag-of-words representation of the items to predict. Same format as the one passed to '.fit'.
		maxiter : int
			Maximum number of iterations for which to run the inference procedure.
		ncores : int
			Number of threads/cores to use. With data for only one user, it's unlikely that using
			multiple threads would give a significant speed-up, and it might even end up making
			the function slower due to the overhead.
			If passing -1, it will determine the maximum number of cores in the system and use that.
		random_seed : int
			Random seed used to initialize parameters.
		stop_thr : float
			If the l2-norm of the difference between values of Theta_{i} between interations is less
			than this, it will stop. Smaller values of 'k' should require smaller thresholds.
		return_all : bool
			Whether to return also the intermediate calculations (Theta_shp, Theta_rte). When
			passing True here, the output will be a tuple containing (Theta, Theta_shp, Theta_rte, Phi)

		Returns
		-------
		factors : array (nitems, k)
			Obtained latent factors/topics for these items.
		"""
		new_Theta_shp, temp = self._predict_item_factors(
								words_df=words_df, maxiter=maxiter, ncores=ncores,
								random_seed=random_seed, stop_thr=stop_thr,
								return_ix=False, return_temp=return_all
								)
		new_Theta_shp = new_Theta_shp / self.Theta_rte

		if return_all:
			return new_Theta_shp, temp
		else:
			return new_Theta_shp

	def _predict_item_factors(self, words_df, maxiter=10, ncores=1, random_seed=1, stop_thr=1e-3,
							  return_ix=True, return_temp=False):
		ncores, maxiter, stop_thr, random_seed = self._process_pars_factors(ncores, maxiter, stop_thr, random_seed, err_subj="item")

		words_df, new_item_mapping = self._process_extra_df(words_df, ttl='words_df')
		words_df['ItemId'] -= self.nitems
		new_max_id = words_df.ItemId.max() + 1
		if new_max_id <= 0:
			raise ValueError("Numeration of item IDs overlaps with IDs passed to '.fit'.")

		new_Theta_shp, temp = cy.calc_item_factors(
					words_df, new_max_id, maxiter, self.k, stop_thr, random_seed, ncores,
					cython_loops.cast_float(self.a), cython_loops.cast_float(self.b),
					cython_loops.cast_float(self.c), cython_loops.cast_float(self.d),
					self.Theta_rte, self.Beta_shp, self.Beta_rte
					)

		if np.isnan(new_Theta_shp).sum().sum() > 0:
			raise ValueError("NaNs encountered in result. Failed to produce latent factors.")

		if self.rescale_factors:
			new_Theta_shp /= new_Theta_shp.sum(axis=1, keepdims=True)

		if return_ix:
			return new_Theta_shp, new_item_mapping, new_max_id

		if return_temp:
			return new_Theta_shp, temp
		else:
			return new_Theta_shp, None

	def predict_user_factors(self, user_df, maxiter=10, ncores=1, random_seed=1, stop_thr=1e-3, return_all=False):
		"""
		Obtain latent factors/topics for users given their attributes alone

		Note
		----
		For better results, refit the model again including these users.

		Note
		----
		If passing more than one user, the resulting rows will be in the sorted order of
		the user IDs from user_df (e.g. if users are 'b', 'a', 'c', the first row will contain
		the factors for user 'a', second for 'b', third for 'c'.

		Note
		----
		This function is prone to producing all NaNs values.

		Parameters
		----------
		user_df : DataFrame (n_samples, 3)
			Attributes of the items to predict. Same format as the one passed to '.fit'.
		maxiter : int
			Maximum number of iterations for which to run the inference procedure.
		ncores : int
			Number of threads/cores to use. With data for only one user, it's unlikely that using
			multiple threads would give a significant speed-up, and it might even end up making
			the function slower due to the overhead.
			If passing -1, it will determine the maximum number of cores in the system and use that.
		random_seed : int
			Random seed used to initialize parameters.
		stop_thr : float
			If the l2-norm of the difference between values of Theta_{i} between interations is less
			than this, it will stop. Smaller values of 'k' should require smaller thresholds.
		return_all : bool
			Whether to return also the intermediate calculations (Z). When
			passing True here, the output will be a tuple containing (Theta_shp, Z)
		only_shape : bool
			Whether to return only the shape parameter for Theta, instead of dividing it by the
			rate parameter.

		Returns
		-------
		factors : array (nitems, k)
			Obtained latent factors/topics for these items.
		"""
		if not self._has_user_df:
			raise ValueError("Can only generate user factors from attributes when the model is fit to user attributes.")

		new_Omega_shp, temp = self._predict_item_factors(
								words_df=user_df, maxiter=maxiter, ncores=ncores,
								random_seed=random_seed, stop_thr=stop_thr,
								return_ix=False, return_temp=return_all
								)
		new_Omega_shp = new_Omega_shp / self.Omega_rte
		if return_all:
			return new_Omega_shp, temp
		else:
			return new_Omega_shp


	def _predict_user_factors(self, user_df, maxiter=10, ncores=1, random_seed=1, stop_thr=1e-3,
							  return_ix=True, return_temp=False):
		ncores, maxiter, stop_thr, random_seed = self._process_pars_factors(ncores, maxiter, stop_thr, random_seed, err_subj="user")
		user_df, new_user_mapping = self._process_extra_df(user_df, ttl='user_df')
		user_df['UserId'] -= self.nusers
		new_max_id = user_df.UserId.max() + 1
		if new_max_id <= 0:
			raise ValueError("Numeration of item IDs overlaps with IDs passed to '.fit'.")

		## Will reuse the exact same function from the add items, need to change names accordingly
		user_df.rename(columns={'UserId':'ItemId', 'AttributeId':'WordId'}, inplace=True)

		new_Omega_shp, temp = cy.calc_item_factors(
					user_df, new_max_id, maxiter, cython_loops.cast_int(self.k),
					stop_thr, random_seed, ncores,
					cython_loops.cast_float(self.a), cython_loops.cast_float(self.b),
					cython_loops.cast_float(self.c), cython_loops.cast_float(self.d),
					self.Omega_rte, self.Kappa_shp, self.Kappa_rte
					)

		if np.isnan(new_Omega_shp).sum().sum() > 0:
			raise ValueError("NaNs encountered in result. Failed to produce latent factors.")

		if self.rescale_factors:
			new_Omega_shp /= new_Omega_shp.sum(axis=1, keepdims=True)

		if return_ix:
			user_df.rename(columns={'ItemId':'UserId', 'WordId':'AttributeId'}, inplace=True)
			return new_Omega_shp, new_user_mapping, new_max_id

		if return_temp:
			return new_Omega_shp, temp
		else:
			return new_Omega_shp, None

	def _process_pars_factors(self, ncores, maxiter, stop_thr, random_seed, err_subj="user"):
		if self.rescale_factors:
			raise ValueError("Cannot produce new factors when using 'rescale_factors=True'.")
		assert self.is_fitted
		if not self.keep_all_objs:
			msg = "Can only add " + err_subj + "s to a fitted model when called with 'keep_all_objs=True'."
			raise ValueError(msg)

		if ncores == -1:
			ncores = multiprocessing.cpu_count()
			if ncores is None:
				ncores = 1 
		assert ncores>0
		assert isinstance(ncores, int)
		ncores = cython_loops.cast_int(ncores)

		assert maxiter > 0
		if isinstance(maxiter, float):
			maxiter = int(maxiter)
		assert isinstance(maxiter, int)
		maxiter = cython_loops.cast_int(maxiter)

		assert stop_thr > 0
		assert isinstance(stop_thr, float)
		stop_thr = cython_loops.cast_float(stop_thr)

		if random_seed is not None:
			if isinstance(random_seed, float):
				random_seed = int(random_seed)
			assert random_seed > 0
			assert isinstance(random_seed, int)

		return ncores, maxiter, stop_thr, random_seed

	def add_items(self, words_df, maxiter=10, stop_thr=1e-3, ncores=1, random_seed=10):
		"""
		Adds new items to an already fit model

		Adds new items without refitting the model from scratch. Note that this will not
		modify any of the user or word parameters.

		For better results, refit the model from scratch including the data from these new items.

		Note
		----
		This function is prone to producing all NaNs values. Adding both users and items to already-fit
		model might cause very bad quality results for both.

		Parameters
		----------
		words_df : data frame or array (n_samples, 3)
			DataFrame with the bag-of-words representation of the new items only. Must contain
			columns 'ItemId', 'WordId', 'Count'. If passing a numpy array, columns will be assumed
			to be in that order.
			When using 'reindex=False', the numeration must start right after the last item ID that
			was present in the training data.
		maxiter : int
			Maximum number of iterations for which to run the procedure.
		stop_thr : float
			Will stop if the norm of the difference between the shape parameters after an iteration
			is below this threshold.
		ncores : int
			Number of threads/core to use. When there is few data, it's unlikely that using
			multiple threads would give a significant speed-up, and it might even end up making
			the function slower due to the overhead.
		random_seed : int or None:
			Random seed to be used for the initialization of the new shape parameters.

		Returns
		-------
		True : bool
			Will return True if the procedure terminates successfully.
		"""
		new_Theta_shp, new_item_mapping, new_max_id = self._predict_item_factors(
					words_df=words_df, maxiter=maxiter, ncores=ncores,
					random_seed=random_seed, stop_thr=stop_thr,
					return_ix=True, return_temp=False
					)

		## Adding the new parameters
		new_Theta = new_Theta_shp / self.Theta_rte
		self.Theta_shp = np.r_[self.Theta_shp, new_Theta]
		self.Theta = np.r_[self.Theta, new_Theta_shp]
		self.Epsilon = np.r_[self.Epsilon, np.zeros((new_max_id, self.k), dtype='float32')]
		self.Epsilon_shp = np.r_[self.Epsilon_shp, np.zeros((new_max_id, self.k), dtype='float32')]
		self.Epsilon_rte = np.r_[self.Epsilon_rte, np.zeros((new_max_id, self.k), dtype='float32')]
		self._M2 = np.r_[self._M2, new_Theta]


		## Adding the new IDs
		if self.reindex:
			self.item_mapping_ = new_item_mapping
			if self.produce_dicts:
				for i in range(new_item_mapping.shape[0] - self.nitems):
					self.item_dict_[new_item_mapping[i + self.nitems]] = i + self.nitems
			self.nitems = self.item_mapping_.shape[0]
		else:
			self.nitems += new_max_id

		return True

	def add_users(self, counts_df=None, user_df=None, maxiter=10, stop_thr=1e-3, ncores=1, random_seed=10):
		"""
		Adds new users to an already fit model

		Adds new users without refitting the model from scratch. Note that this will
		not modify any of the item or word parameters. In the regular model, you will
		need to provide "counts_df" as input, and the parameters will be determined
		according to the user-item interactions. If fitting the model with user attributes,
		you will also need to provide "user_df". Not providind a 'counts_df' object will
		assume that all the interactions for this user are zero (only supported in the model
		with user attributes).

		For better results, refit the model from scratch including the data from these new users.

		Note
		----
		This function is prone to producing all NaNs values. Adding both users and items to already-fit
		model might cause very bad quality results for both.

		Parameters
		----------
		counts_df : data frame or array (n_samples, 3)
			DataFrame with the user-item interactios for the new users only. Must contain
			columns 'UserId', 'ItemId', 'Count'. If passing a numpy array, columns will be assumed
			to be in that order.
		user_df : data frame or array (n_samples, 3)
			DataFrame with the user attributes for the new users only. Must contain columns
			'UserId', 'AttributeId', 'Count'. If passing a numpy array, columns will be assumed to be
			in that order. Only for models with to user side information.
		maxiter : int
			Maximum number of iterations for which to run the procedure.
		stop_thr : float
			Will stop if the norm of the difference between the shape parameters after an iteration
			is below this threshold.
		ncores : int
			Number of threads/core to use. When there is few data, it's unlikely that using
			multiple threads would give a significant speed-up, and it might even end up making
			the function slower due to the overhead.
		random_seed : int or None:
			Random seed to be used for the initialization of the new shape parameters.

		Returns
		-------
		True : bool
			Will return True if the procedure terminates successfully.
		"""

		ncores, maxiter, stop_thr, random_seed = self._process_pars_factors(ncores, maxiter, stop_thr, random_seed, err_subj="user")

		## checking input combinations
		if (counts_df is None) and (user_df is None):
			raise ValueError("Must pass at least one of 'counts_df' or 'user_df'.")

		if user_df is not None:
			if not self._has_user_df:
				raise ValueError("Can only use 'user_df' when the model was fit to user side information.")

		if (counts_df is None) and (not self._has_user_df):
			raise ValueError("Must pass 'counts_df' to add a new user.")

		
		if (counts_df is not None) and (user_df is not None) and self._has_user_df:
			## factors based on both attributes and interactions
			user_df, counts_df, new_user_mapping = self._process_extra_df(user_df, ttl='user_df', df2=counts_df)
			counts_df['UserId'] -= self.nusers
			user_df['UserId'] -= self.nusers
			new_max_id = max(counts_df.UserId.max(), user_df.UserId.max()) + 1
			if new_max_id <= 0:
				raise ValueError("Numeration of item IDs overlaps with IDs passed to '.fit'.")

			new_Omega_shp, new_Eta_shp = cy.calc_user_factors_full(
					counts_df, user_df, new_max_id, cython_loops.cast_int(maxiter), cython_loops.cast_int(self.k),
					stop_thr, random_seed, ncores,
					cython_loops.cast_float(self.c), cython_loops.cast_float(self.e),
					self.Omega_rte, self.Eta_rte,
					self.Theta_shp, self.Theta_rte,
					self.Epsilon_shp, self.Epsilon_rte,
					self.Kappa_shp, self.Kappa_rte
				)

			## Adding the new parameters
			new_Omega = new_Omega_shp / self.Omega_rte
			new_Eta = new_Eta_shp / self.Eta_rte
			self.Omega_shp = np.r_[self.Omega_shp, new_Omega_shp]
			self.Omega = np.r_[self.Omega, new_Omega]
			self.Eta_shp = np.r_[self.Eta_shp, new_Eta_shp]
			self.Eta = np.r_[self.Eta, new_Eta]
			self._M1 = np.r_[self._M1, new_Omega + new_Eta]
		
		## factors based on user-item interactions
		elif (user_df is None) and (counts_df is not None):
			
			counts_df, new_user_mapping = self._process_extra_df(counts_df, ttl='counts_df')
			counts_df['UserId'] -= self.nusers
			new_max_id = counts_df.UserId.max() + 1
			if new_max_id <= 0:
				raise ValueError("Numeration of item IDs overlaps with IDs passed to '.fit'.")

			new_Eta_shp = cy.calc_user_factors(
					counts_df, new_max_id, maxiter, cython_loops.cast_int(self.k),
					stop_thr, random_seed, ncores,
					cython_loops.cast_float(self.e), self.Eta_rte,
					self.Theta_shp, self.Theta_rte, self.Epsilon_shp, self.Epsilon_rte
					)

			## Adding the new parameters
			new_Eta = new_Eta_shp / self.Eta_rte
			self.Eta_shp = np.r_[self.Eta_shp, new_Eta_shp]
			self.Eta = np.r_[self.Eta, new_Eta]
			self._M1 = np.r_[self._M1, new_Eta]
			if self._has_user_df:
				self.Omega = np.r_[self.Omega, np.zeros((new_max_id, self.k), dtype='float32')]
				self.Omega_shp = np.r_[self.Omega_shp, np.zeros((new_max_id, self.k), dtype='float32')]

		## factors based on user attributes
		else:
			new_Omega_shp, new_user_mapping, new_max_id = self._predict_user_factors(
					user_df=user_df, maxiter=maxiter, ncores=ncores,
					random_seed=random_seed, stop_thr=stop_thr,
					return_ix=True, return_temp=False
					)

			## Adding the new parameters
			new_Omega = new_Omega_shp / self.Omega_rte
			self.Omega_shp = np.r_[self.Omega_shp, new_Omega_shp]
			self.Omega = np.r_[self.Omega, new_Omega]
			self.Eta = np.r_[self.Eta, np.zeros((new_max_id, self.k), dtype='float32')]
			self.Eta_shp = np.r_[self.Eta_shp, np.zeros((new_max_id, self.k), dtype='float32')]
			self._M1 = np.r_[self._M1, new_Omega]

		
		## updating the list of seen items for these users
		if self.keep_data and (counts_df is not None):
			for u in range(new_max_id):
				items_this_user = counts_df.ItemId.values[counts_df.UserId == u]
				self._n_seen_by_user = np.r_[self._n_seen_by_user, items_this_user.shape[0]]
				self._st_ix_user = np.r_[self._st_ix_user, self.seen.shape[0]]
				self.seen = np.r_[self.seen, items_this_user]
		
		## Adding the new IDs
		if self.reindex:
			self.user_mapping_ = new_user_mapping
			if self.produce_dicts:
				for u in range(new_user_mapping.shape[0] - self.nusers):
					self.user_dict_[new_user_mapping[u + self.nusers]] = u + self.nusers
			self.nusers = self.user_mapping_.shape[0]
		else:
			self.nitems += new_max_id

		return True

	def eval_llk(self, counts_df, full_llk=False):
		"""
		Evaluate Poisson log-likelihood (plus constant) for a given dataset
		
		Note
		----
		This log-likelihood is calculated only for the combinations of users and items
		provided here, so it's not a complete likelihood, and it might sometimes turn out to
		be a positive number because of this.
		Will filter out the input data by taking only combinations of users
		and items that were present in the training set.

		Parameters
		----------
		counts_df : pandas data frame (nobs, 3)
			Input data on which to calculate log-likelihood, consisting of IDs and counts.
			Must contain one row per non-zero observaion, with columns 'UserId', 'ItemId', 'Count'.
			If a numpy array is provided, will assume the first 3 columns
			contain that info.
		full_llk : bool
			Whether to calculate terms of the likelihood that depend on the data but not on the
			parameters. Ommitting them is faster, but it's more likely to result in positive values.

		Returns
		-------
		llk : dict
			Dictionary containing the calculated log-likelihood and the number of
			observations that were used to calculate it.
		"""
		assert self.is_fitted
		HPF._process_valset(self, counts_df, valset=False)
		out = {'llk': cython_loops.calc_llk(self.val_set.Count.values,
											self.val_set.UserId.values,
											self.val_set.ItemId.values,
											self._M1,
											self._M2,
											cython_loops.cast_int(self.k),
											cython_loops.cast_int(self.ncores),
											cython_loops.cast_int(bool(full_llk))),
			   'nobs':self.val_set.shape[0]}
		del self.val_set
		return out

	def _print_st_msg(self):
		print("*****************************************")
		print("Collaborative Topic Poisson Factorization")
		print("*****************************************")
		print("")

	def _print_data_info(self):
		print("Number of users: %d" % self.nusers)
		print("Number of items: %d" % self.nitems)
		print("Number of words: %d" % self.nwords)
		if self._has_user_df:
			print("Number of user attributes: %d" % self.nuserattr)
		print("Latent factors to use: %d" % self.k)
		print("")


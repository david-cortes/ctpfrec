# Collaborative Topic Poisson Factorization

Python implementation of the algorithm for probabilistic matrix factorization described in _Content-based recommendations with poisson factorization (Gopalan, P.K., Charlin, L. and Blei, D., 2014)_.

This is a statistical model aimed at recommender systems with implicit data consisting of counts of user-item interactions (e.g. clicks by each user on different products) plus bag-of-words representations of the items. The model is fit using mean-field variational inference. Can also fit the model to side information on the users consisting of counts on different attributes (same format as the bag-of-words for items).

As it takes side information about items, it has the advantage of being able to recommend items without any ratings/clicks/plays/etc. If extending it with user side information, can also make cold-start recommendations, albeit speed is not great for that.

Supports parallelization, different stopping criteria for the optimziation procedure, and adding users/items without refitting the model entirely. The bottleneck computations are written in fast Cython code.

For a similar package for explicit feedback data see also [cmfrec](https://github.com/david-cortes/cmfrec). For Poisson factorization without side information see [hpfrec](https://github.com/david-cortes/hpfrec).

## Model description

The model consists in producing non-negative low-rank matrix factorizations of counts data (such as number of times each user played each song in some internet service) of user-item interactions and item-word counts, produced by a generative model specified as follows:

```
Item model:
B_vk ~ Gamma(a, b)
T_ik ~ Gamma(c, d)
W_iv ~ Poisson(T * B')

Interactions model:
N_uk ~ Gamma(e, f)
E_ik ~ Gamma(g, h)
R_ui ~ Poisson(N * (T + E)')
```

_(Where `W` is the bag-of-words representation of the items, `R` is the user-item interactions matrix, `u` is the number of users, `i` is the number of items, `v` is the number of words, and `k` is the number of latent factors or topics)_

For more details see the references section at the bottom.

When adding user information, the model becomes as follows:

```
Item model:
B_vk ~ Gamma(a, b)
T_ik ~ Gamma(c, d)
W_iv ~ Poisson(T * B')

User model:
K_ak ~ Gamma(e, f)
O_uk ~ Gamma(l, m)
Q_ua ~ Poisson(O * K')

Interactions model:
N_uk ~ Gamma(i, j)
E_ik ~ Gamma(g, h)
R_ui ~ Poisson((O + N) * (T + E)')
```

A huge drawback of this model compared to LDA is that, as the matrices are non-negative, items with more words will have larger values in their factors/topics, which will result in them having higher scores regardless of their popularity. This effect can be somewhat decreased by using only a limited number of words to represent each item (scaling upwards the ones that don't have enough words), by standardizing the bag-of-words to have all rows summing up to a certain number (this is hard to do when the counts are supposed to be integers, but the package can still work mostly fine with decimals that are at least >= 0.9, and has the option to standardize the inputs), or to a lesser extent by standardizing the resulting Theta shape matrix to have its rows sum to 1 (also supported in the package options).

#### Why is it more efficient

In typical settings for recommendations with implicit data, most users ever see/click/play/buy a handful selected items out of all the available catalog, thus a matrix of user-item interactions would be extremely sparse (most entries would be zero). Algorithms like implicit-ALS or BPR (Bayesian personalized ranking) require iterating over some or all of the missing combinations not seen in the data (e.g. songs not played by each user) in order to compute their respective loss functions, which is slow and not very scalable.

However, Poisson likelihood is given by the formula:
```L(y) = yhat^y * exp(-yhat) / y!```

If taking the logarithm (log-likelihood), then this becomes:
```l(y) = -log(y!) + y*log(yhat) - yhat```

Since `log(0!) = 0`, `0*log(yhat) = 0`, and the sum of predictions for all combinations of users and items can be quickly calculated by `sum yhat = sum_{i,j} <U_i, V_j> = <sum_i U_i, sum_j V_j>` (since `U` and `V` are non-negative matrices), it means the model doesn't ever need to make calculations on values that are equal to zero in order to determine their Poisson log-likelihood.

Moreover, negative Poisson log-likelihood is a more appropriate loss for count data than squared loss, which tends to produce not-so-good results when the values to predict follow an exponential rather than a normal distribution.

## Installation

Package is available on PyPI, can be installed with

```pip install ctpfrec```

As it contains Cython code, it requires a C compiler. In Windows, this usually means it requires a Visual Studio Build Tools installation with MSVC140 (or MinGW + GCC), otherwise the installation from `pip` might fail. For more details see this guide:
[Cython Extensions On Windows](https://github.com/cython/cython/wiki/CythonExtensionsOnWindows)

On Linux, the `pip` install should work out-of-the-box, as long as the system has `gcc`.

On Mac, installing this package will first require setting up OpenMP for `clang` compiler (does not come by default in apple's redistributions), or installing `gcc`.

The package has only been tested under Python 3.6 and **some functionalities will not work on Python 2.7**.

## Sample usage

```python
import numpy as np, pandas as pd
from ctpfrec import CTPF

## Generating a fake dataset
nusers = 10**2
nitems = 10**2
nwords = 5 * 10**2
nobs = 10**4
nobs_bag_of_words = 10**4

np.random.seed(1)
counts_df = pd.DataFrame({
	'UserId' : np.random.randint(nusers, size=nobs),
	'ItemId' : np.random.randint(nitems, size=nobs),
	'Count' : np.random.gamma(1, 1, size=nobs).astype('int32')
	})
counts_df = counts_df.loc[counts_df.Count > 0]

words_df = pd.DataFrame({
	'ItemId' : np.random.randint(nitems, size=nobs_bag_of_words),
	'WordId' : np.random.randint(nwords, size=nobs_bag_of_words),
	'Count' : np.random.gamma(1, 1, size=nobs_bag_of_words).astype('int32')
	})
words_df = words_df.loc[words_df.Count > 0]

## Fitting the model
recommender = CTPF(k = 15, reindex=True)
recommender.fit(counts_df=counts_df, words_df=words_df)

## Making predictions
recommender.topN(user=10, n=10, exclude_seen=True)
recommender.topN(user=10, n=10, exclude_seen=False, items_pool=np.array([1,2,3,4]))
recommender.predict(user=10, item=11)
recommender.predict(user=[10,10,10], item=[1,2,3])
recommender.predict(user=[10,11,12], item=[4,5,6])

## Evaluating Poisson log-likelihood
recommender.eval_llk(counts_df, full_llk=True)

## Adding new items without refitting
nitems_new = 10
nobs_bow_new = 2 * 10**3
np.random.seed(5)
words_df_new = pd.DataFrame({
	'ItemId' : np.random.uniform(low=nitems, high=nitems+nitems_new, size=nobs_bow_new),
	'WordId' : np.random.randint(nwords, size=nobs_bow_new),
	'Count' : np.random.gamma(1, 1, size=nobs_bow_new).astype('int32')
	})
words_df_new = words_df_new.loc[words_df_new.Count > 0]

recommender.add_items(words_df_new)
```

If passing `reindex=True`, all user and item IDs that you pass to `.fit` will be reindexed internally (they need to be hashable types like `str`, `int` or `tuple`), and you  can use these same IDs to make predictions later. The IDs returned by `topN` are these same IDs passed to `.fit` too.

For a more detailed example, see the IPython notebook [recommending products with RetailRocket's event logs](http://nbviewer.jupyter.org/github/david-cortes/ctpfrec/blob/master/example/ctpfrec_retailrocket.ipynb) illustrating its usage with the RetailRocket dataset consisting of activity logs (view, add-to-basket, purchase) and item descriptions.

This package contains only functionality related to fitting this model. For general evaluation metrics for recommendations on implicit data see other packages such as [lightFM](https://github.com/lyst/lightfm).

## Documentation

Documentation is available at readthedocs: [http://ctpfrec.readthedocs.io](http://ctpfrec.readthedocs.io/en/latest/)

It is also internally documented through docstrings (e.g. you can try `help(ctpfrec.CTPF))`, `help(ctpfrec.CTPF.fit)`, etc.

## Speeding up optimization procedure

For faster fitting and predictions, use SciPy and NumPy libraries compiled against MKL. In Windows, you can find Python wheels (installable with pip after downloading them) of numpy and scipy precompiled with MKL in [Christoph Gohlke's website](https://www.lfd.uci.edu/~gohlke/pythonlibs/). In Linux and Mac, these come by default in Anaconda installations (but are likely to get overwritten if you enable `conda-forge`).

The constructor for CTPF allows some parameters to make it run faster (if you know what you're doing): these are `allow_inconsistent_math=True`, `full_llk=False`, `stop_crit='diff-norm'`, `reindex=False`, `verbose=False`. See the documentation for more details.

## Saving model with pickle

Don't use `pickle` to save an `CTPF` object, as it will fail due to problems with lambda functions. Use `dill` instead, which has the same syntax as pickle:

```python
import dill
from ctpfrec import CTPF

c = CTPF()
dill.dump(c, open("CTPF_obj.dill", "wb"))
c = dill.load(open("CTPF_obj.dill", "rb"))
```

## References
[1] Gopalan, Prem K., Laurent Charlin, and David Blei. "Content-based recommendations with poisson factorization." Advances in Neural Information Processing Systems. 2014.

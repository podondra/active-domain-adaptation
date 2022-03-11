# Model uncertainty in active domain adaptation

- TODO: Write introduction.
- TODO: Test or verify code.

Abbreviations:
- active domain adaptation (ADA);
- convolutional neural network (CNN);
- domain-adversarial neural network (DANN);
- principal component analysis (PCA);
- quasar (QSO);
- Street View House Numbers (SVHN);
- Sloan Digital Sky Survey (SDSS);
- t-Distributed Stochastic Neighbor Embedding (t-SNE);
- Uniform Manifold Approximation and Projection (UMAP).

Hypotheses:
1. Deep ensembles are robust to dataset shift so they will improve ADA.
1. Astronomical data have properties that will reveal problems of ADA methods.
1. Quering data for labelling from a visualisation of data with reduced dimensionality (e.g. by UMAP) will achieve diversity of labelled data and avoid problems of ADA methods.
1. Methods with good predictive uncertainty firstly query outliers so they cannot performance stagnates when the amount of labelled data is not sufficient. 

## Domain adaptation

- TODO: Formalise according to Redko et al. (2019): (source and target) domain, task etc.

## Deep active learning

## Active domain adapation

It is important to query both 1. *uncertain* and 2. *diverse* data for labelling (probably Prabhu et al. 2021).

## Related work

- TODO: Read relate work and implement some methods as baselines.

## Proposed method

- TODO: Distinguish kinds of uncertainties: predictive, data (aleatoric), model (epistemic).
- TODO: We use UMAP. However, there are other methods: t-SNE, PCA etc. Which one to choose and what are the arguments?
- TODO: Reduce dimensionality of original data or data embedding? Prabhu et al. (2021) uses embedding).

Deep machine learning model (deep model) that is convolutional neural network (CNN) to process data.
However, the CNN will produce predictive uncertainty to get model uncertainty.
There are two main methods how to get predictive uncertainty:
- MC dropout (Gal 2016);
- deep ensemble (Lakshminarayanan et al. 2017).
Ovadia et al. (2019) compares methods how to get predictive uncertainty and conclude:
"Deep ensembles seem to perform the best across most metrics and be more robust to dataset shift." (p. 9)
We are interested in the feature of deep ensembles that they are robust to dataset shift that should be crucial for domain adaptation.
Deep ensembles will make sure that we query uncetaint data for labelling.

However, we also have to query diverse data for labelling.
We will use a visualisation of data with reduced dimensionality to query data for labelling.
We can query diverse sample from the visualisation.
Moreover, we can avoid quering outliers (the potential problem of methods that provide good predictive uncertainty).

Free parameters (either provide arguments or experiments):
1. query size;
1. learning strategy (Prabhu et al. (2021) experimented with fine-tuning (probably Yosinski et al. 2014), MME (Saito et al. 2019)), DANN (Ganin et al. 2016) and found MME to be the best);
1. the amout of data in the visualisation;
1. dimensionality reduction method (t-SNE, UMAP, PCA etc.).

What we found:
1. The amount of data in the visualisation has to be high enoungh else the query will not be diverse enough.

## Experiments

The proposed method is time consuming because we cannot simulate it.
We have to query data manually.
These are the argument why to experiment only with 1. SVHN and MNIST, 2. SDSS QSO catalogues.

### Classification from SVHN to MNIST

- TODO: Describe the variant of LeNet.
- TODO Should I do data preparation, e.g. stadardisation or normalisation?

SVHN dataset is available online, at http://ufldl.stanford.edu/housenumbers/.

### Redshift prediction from SDSS DR12Q to DR16Q superset

Task:
I will used a variant of SZNet to predict spectroscopic redshift.
Training data (source domain) are spectra from the superset of SDSS data release (DR) 12 QSO catalogue.

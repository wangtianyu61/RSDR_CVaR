This repo is the codes to [Data-Driven Distributionally Robust CVaR Portfolio Optimization Under A Regime-Switching Ambiguity Set](https://pubsonline.informs.org/doi/10.1287/msom.2023.1229).


The overall pipeline document and guidance is in the ``code/portfolio_experiment_demiguel.ipynb`` (existing dataset) and ``portfolio_experiment_final_version.ipynb`` (our constructed dataset). Detailed user guides are listed in the corresponding notebook files.

We have sample data in the ``factor model`` folder. Our constructed dataset is ``factor model/16_IndustryIndices_Monthly.csv``. Other data used in our paper can be downloaded from Kenneth French website and transformed into the same ``.csv`` format. For the regimes data, we have processed through other tools and the regimes estimated from bull and bear, weathers and hidden markov data in the ``*.state.csv``. Note that the data is up until 04-2019. Follow-up data can be extracted following our instructions.




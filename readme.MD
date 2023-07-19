This repo is the codes to [Data-Driven Distributionally Robust CVaR Portfolio Optimization Under A Regime-Switching Ambiguity Set](https://pubsonline.informs.org/doi/10.1287/msom.2023.1229).


We have sample data in the ``factor model`` folder. Our constructed dataset is ``factor model/16_IndustryIndices_Monthly.csv``. Other data used in our paper can be downloaded from [Kenneth French website](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) and transformed into the same ``.csv`` format. Other data and baseline methods can be extracted from that website and directly compatible with the current codebase.

For the regimes data, we have processed through other tools and the regimes estimated from bull and bear, weathers and hidden markov data in the ``*.state.csv``. Note that the data is up until 04-2019. Follow-up data can be extracted following our instructions in the paper.

The overall pipeline document and guidance is in the ``code/portfolio_experiment_demiguel.ipynb`` (existing dataset) and ``portfolio_experiment_final_version.ipynb`` (our constructed dataset). Detailed user guides are listed in the corresponding notebook files.

To output the summary statistics shown in the paper, a folder named ``output\`` (or other name) need to be created under the root directory to store the output data when running the output code as long as the the folder of csv_name exists. (Currently the folder name is ``output`` in the two notebooks).




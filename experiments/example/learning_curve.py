import os
import sys
sys.path.append(os.getcwd() + '/src')

import numpy as np
import matplotlib.pyplot as plt
from PyExpPlotting.matplot import save, setDefaultConference
from PyExpUtils.results.Collection import ResultCollection

from RlEvaluation.config import data_definition
from RlEvaluation.temporal import TimeSummary, extract_multiple_learning_curves, curve_percentile_bootstrap_ci
from RlEvaluation.statistics import Statistic
from RlEvaluation.utils.pandas import split_over_column

import RlEvaluation.hypers as Hypers

# from analysis.confidence_intervals import bootstrapCI
from experiment.ExperimentModel import ExperimentModel
from experiment.tools import parseCmdLineArgs

# makes sure figures are right size for the paper/column widths
# also sets fonts to be right size when saving
setDefaultConference('jmlr')

COLORS = {
    'TDRC': 'purple',
    'TDC': 'blue',
    'GTD2': 'red',
    'TD': 'black',
    'HTD': 'green',
    'VTrace': 'yellow',
}

# keep 1 in every SUBSAMPLE measurements
SUBSAMPLE = 1

if __name__ == "__main__":
    path, should_save, save_type = parseCmdLineArgs()

    results = ResultCollection.fromExperiments(Model=ExperimentModel)

    data_definition(
        hyper_cols=results.get_hyperparameter_columns(),
        seed_col='seed',
        time_col='frame',
        environment_col='environment',
        algorithm_col='algorithm',

        # makes this data definition globally accessible
        # so we don't need to supply it to all API calls
        make_global=True,
    )

    df = results.combine(
        # converts path like "experiments/example/MountainCar"
        # into a new column "environment" with value "MountainCar"
        # None means to ignore a path part
        folder_columns=(None, None, 'environment'),

        # and creates a new column named "algorithm"
        # whose value is the name of an experiment file, minus extension.
        # For instance, ESARSA.json becomes ESARSA
        file_col='algorithm',
    )

    assert df is not None

    exp = results.get_any_exp()

    for metric in ['mspbe', 'msve']:
        for env, env_df in split_over_column(df, col='environment'):
            f, ax = plt.subplots()
            print('-' * 50)
            for alg, sub_df in split_over_column(env_df, col='algorithm'):
                if len(sub_df) == 0: continue

                report = Hypers.select_best_hypers(
                    sub_df,
                    metric='mspbe',
                    prefer=Hypers.Preference.low,
                    time_summary=TimeSummary.mean,
                    statistic=Statistic.mean,
                )

                print('-' * 25)
                print(env, alg, metric)
                Hypers.pretty_print(report)

                xs, ys = extract_multiple_learning_curves(
                    sub_df,
                    report.uncertainty_set_configurations,
                    metric=metric,
                )

                xs = np.asarray(xs)[0, ::SUBSAMPLE]
                ys = np.asarray(ys)[:, ::SUBSAMPLE]
                ys = np.sqrt(ys)

                res = curve_percentile_bootstrap_ci(
                    rng=np.random.default_rng(0),
                    y=ys,
                    statistic=Statistic.mean.value,
                )

                ax.plot(xs, res.sample_stat, label=alg, color=COLORS[alg], linewidth=0.5)
                ax.fill_between(xs, res.ci[0], res.ci[1], color=COLORS[alg], alpha=0.2)

            ax.legend()
            if should_save:
                save(
                    save_path=f'{path}/plots',
                    plot_name=f'{env}.r{metric}'
                )
                plt.clf()
            else:
                plt.show()
                exit()

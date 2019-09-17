package experiments.two_part.part_two;

import ranker.core.algorithms.Ranker;
import ranker.core.algorithms.decomposition.rankprediction.RankRegressionRanker;

public class RankRegressionRankerExperimenter extends RankerExecutor {

	@Override
	protected Class<? extends RankerConfig> getRankerConfigClass() {
		return WEKARegressionRankerConfig.class;
	}

	@Override
	protected Ranker instantiate(RankerConfig configuration) {
		WEKARegressionRankerConfig config = (WEKARegressionRankerConfig) configuration;

		return new RankRegressionRanker(config.getAlgorithm(), config.getHyperparameters());
	}

}

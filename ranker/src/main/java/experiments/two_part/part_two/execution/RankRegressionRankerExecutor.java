package experiments.two_part.part_two.execution;

import ranker.core.algorithms.Ranker;
import ranker.core.algorithms.decomposition.rankprediction.RankRegressionRanker;

public class RankRegressionRankerExecutor extends RankerExecutor {

	@Override
	protected Class<? extends RankerConfig> getRankerConfigClass() {
		return RankRegressionRankerConfig.class;
	}

	@Override
	protected Ranker instantiate(RankerConfig configuration) {
		RankRegressionRankerConfig config = (RankRegressionRankerConfig) configuration;

		return new RankRegressionRanker(config.getRegressorName(), config.getRegressorOptions());
	}
	
	public static void main (String [] args) throws Exception {
		new RankRegressionRankerExecutor().evaluateRankerWithArgs(args);
	}

	@Override
	protected String getActiveConfiguration() {
		return "";
	}
}

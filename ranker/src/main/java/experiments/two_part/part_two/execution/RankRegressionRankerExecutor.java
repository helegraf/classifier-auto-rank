package experiments.two_part.part_two.execution;

import java.util.List;

import ranker.core.algorithms.Ranker;
import ranker.core.algorithms.decomposition.rankprediction.RankRegressionRanker;
import weka.core.Instances;

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
	
	@Override
	protected String getActiveConfiguration() {
		//TODO
		return "";
	}

	@Override
	protected Ranker getOptimalRanker(Instances hyperTrain, Instances hyperTest, List<Integer> targetAttributes,
			RankerConfig configuration) {
		// TODO Auto-generated method stub
		return null;
	}

	public static void main (String [] args) throws Exception {
		new RankRegressionRankerExecutor().evaluateRankerWithArgs(args);
	}
}

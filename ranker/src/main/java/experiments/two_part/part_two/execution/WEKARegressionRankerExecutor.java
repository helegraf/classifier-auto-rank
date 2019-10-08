package experiments.two_part.part_two.execution;

import java.util.List;

import ranker.core.algorithms.Ranker;
import ranker.core.algorithms.decomposition.regression.WEKARegressionRanker;
import weka.core.Instances;

public class WEKARegressionRankerExecutor extends RankerExecutor {

	@Override
	protected Class<? extends RankerConfig> getRankerConfigClass() {
		return WEKARegressionRankerConfig.class;
	}

	@Override
	protected Ranker instantiate(RankerConfig configuration) {
		WEKARegressionRankerConfig config = (WEKARegressionRankerConfig) configuration;
		
		return new WEKARegressionRanker(config.getAlgorithm(), config.getHyperparameter());
	}
	
	@Override
	protected String getActiveConfiguration() {
		// TODO Auto-generated method stub
		return "";
	}

	@Override
	protected Ranker getOptimalRanker(Instances hyperTrain, Instances hyperTest, List<Integer> targetAttributes,
			RankerConfig configuration) {
		// TODO Auto-generated method stub
		return null;
	}

	public static void main (String [] args) throws Exception {
		new WEKARegressionRankerExecutor().evaluateRankerWithArgs(args);
	}

}

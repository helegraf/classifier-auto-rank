package experiments.two_part.part_two.execution;

import ranker.core.algorithms.Ranker;
import ranker.core.algorithms.decomposition.regression.WEKARegressionRanker;

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
	
	public static void main (String [] args) throws Exception {
		new WEKARegressionRankerExecutor().evaluateRankerWithArgs(args);
	}

	@Override
	protected String getActiveConfiguration() {
		// TODO Auto-generated method stub
		return "";
	}
}

package experiments.two_part.part_two.execution;

import java.util.List;

import ranker.core.algorithms.Ranker;
import ranker.core.algorithms.baseline.BestAlgorithmRanker;
import weka.core.Instances;

public class BestAlgorithmRankerExecutor extends RankerExecutor {

	@Override
	protected Class<? extends RankerConfig> getRankerConfigClass() {
		return RankerConfig.class;
	}

	@Override
	protected Ranker instantiate(RankerConfig configuration) {
		return new BestAlgorithmRanker();
	}
	
	public static void main(String [] args) throws Exception {
		new BestAlgorithmRankerExecutor().evaluateRankerWithArgs(args);
	}

	@Override
	protected String getActiveConfiguration() {
		return "";
	}

	@Override
	protected Ranker getOptimalRanker(Instances hyperTrain, Instances hyperTest, List<Integer> targetAttributes,
			RankerConfig configuration) {
		return this.instantiate(configuration);
	}

}

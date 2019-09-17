package experiments.two_part.part_two;

import ranker.core.algorithms.Ranker;
import ranker.core.algorithms.baseline.BestAlgorithmRanker;

public class BestAlgorithmRankerExecutor extends RankerExecutor {

	@Override
	protected Class<? extends RankerConfig> getRankerConfigClass() {
		return RankerConfig.class;
	}

	@Override
	protected Ranker instantiate(RankerConfig configuration) {
		return new BestAlgorithmRanker();
	}

}

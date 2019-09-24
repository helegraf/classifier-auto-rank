package experiments.two_part.part_two.execution;

import ranker.core.algorithms.Ranker;
import ranker.core.algorithms.decomposition.conflictresolution.ExpectedRankStrategy;
import ranker.core.algorithms.decomposition.conflictresolution.HighestProbabilityStrategy;
import ranker.core.algorithms.decomposition.rankprediction.RankClassificationRanker;

public class RankClassificationRankerExecutor extends RankerExecutor {

	@Override
	protected Class<? extends RankerConfig> getRankerConfigClass() {
		return RankClassificationRankerConfig.class;
	}

	@Override
	protected Ranker instantiate(RankerConfig configuration) {
		RankClassificationRankerConfig config = (RankClassificationRankerConfig) configuration;

		RankClassificationRanker ranker = new RankClassificationRanker(config.getClassifierName(), config.getClassifierOptions());

		switch (config.getConflictResolutionStrategy()) {
		case "ExpectedRankStrategy":
			ranker.setConflictResolutionStrategy(new ExpectedRankStrategy());
			break;
		case "HighestProbabilityStrategy":
			ranker.setConflictResolutionStrategy(new HighestProbabilityStrategy());
			break;
		default:
			throw new IllegalArgumentException(
					String.format("%s is not a valid conflict resolution strategy for the rank classification ranker.",
							config.getConflictResolutionStrategy()));
		}
		
		return ranker;
	}
	
	public static void main (String [] args) throws Exception {
		new RankClassificationRankerExecutor().evaluateRankerWithArgs(args);
	}

	@Override
	protected String getActiveConfiguration() {
		// TODO Auto-generated method stub
		return "";
	}

}

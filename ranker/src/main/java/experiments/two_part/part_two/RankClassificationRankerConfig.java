package experiments.two_part.part_two;

public interface RankClassificationRankerConfig extends WEKARegressionRankerConfig {
	
	@Key("rankclassificationranker.conflictresolution")
	@DefaultValue("ExpectedRankStrategy")
	public String getConflictResolutionStrategy();
}

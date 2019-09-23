package experiments.two_part.part_two.execution;

public interface RankClassificationRankerConfig extends RankerConfig {
	
	@Key("rankclassificationranker.conflictresolution")
	@DefaultValue("ExpectedRankStrategy")
	public String getConflictResolutionStrategy();
	
	@Key("rankclassificationranker.classifier")
	@DefaultValue("weka.classifiers.trees.RandomForest")
	public String getClassifierName();
	
	@Key("rankclassificationranker.classifier.options")
	@DefaultValue("-P, 100, -I, 100, -num-slots, 1, -K, 0, -M, 1.0, -V, 0.001, -S, 1")
	public String [] getClassifierOptions();
}

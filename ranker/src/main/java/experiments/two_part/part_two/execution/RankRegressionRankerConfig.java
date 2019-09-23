package experiments.two_part.part_two.execution;

public interface RankRegressionRankerConfig extends RankerConfig {
	
	@Key("rankclassificationranker.regressor")
	@DefaultValue("weka.classifiers.trees.RandomForest")
	public String getRegressorName();
	
	@Key("rankclassificationranker.regressor.options")
	@DefaultValue("-P, 100, -I, 100, -num-slots, 1, -K, 0, -M, 1.0, -V, 0.001, -S, 1")
	public String [] getRegressorOptions();

}

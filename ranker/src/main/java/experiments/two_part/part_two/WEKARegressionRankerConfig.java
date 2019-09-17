package experiments.two_part.part_two;

import java.util.Arrays;

public interface WEKARegressionRankerConfig extends RankerConfig {

	@Key("wekaranker.algorithm")
	@DefaultValue("weka.classifiers.trees.RandomForest")
	public String getAlgorithm();

	@Key("wekaranker.hyperparameters")
	public String getHyperparameterString();

	public default String[] getHyperparameters() {
		String[] hyperparameters = null;
		if (this.getHyperparameterString() != null) {
			String params = this.getHyperparameterString().substring(1, this.getHyperparameterString().length() - 1);
			hyperparameters = Arrays.stream(params.split(",")).toArray(String[]::new);
		}

		return hyperparameters;
	}
}

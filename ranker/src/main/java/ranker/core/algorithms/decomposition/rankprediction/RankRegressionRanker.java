package ranker.core.algorithms.decomposition.rankprediction;

import java.util.Arrays;
import java.util.List;

import ranker.core.algorithms.decomposition.regression.WEKARegressionRanker;
import weka.core.Instance;

/**
 * Predicts a ranking of algorithms by predicting the rank of each algorithm
 * with a regression model. To this end, performance data in the training data
 * is replaced by rank information.
 * 
 * @author helegraf
 *
 */
public class RankRegressionRanker extends WEKARegressionRanker {

	/**
	 * Construct a RankRegressionRanker with the given name (fully qualified name of
	 * a weka classifier which will be used for predictions).
	 * 
	 * @param name fully qualified name of a weka classifier
	 */
	public RankRegressionRanker(String name) {
		super(name);
	}
	
	public RankRegressionRanker(String name, String[] hyperparameters) {
		super(name, hyperparameters);
	}

	@Override
	protected void modifyInstance(Instance instance, List<Integer> targetAttributes) {
		// for each instance, the performance of the classifiers is substituted with
		// their rank
		// get target value values
		double[] targetValues = new double[targetAttributes.size()];
		for (int i = 0; i < targetAttributes.size(); i++) {
			targetValues[i] = instance.value(targetAttributes.get(i));
		}

		// sort (in ascending order (worst first) so that the LOWEST predicted rank
		// later will be the LAST in the ranking
		Arrays.sort(targetValues);

		// go through the values and replace with ranks
		for (int i = 0; i < targetAttributes.size(); i++) {
			double previousValue = instance.value(targetAttributes.get(i));

			for (int j = 0; j < targetValues.length; j++) {
				if (targetValues[j] == previousValue) {
					instance.setValue(targetAttributes.get(i), j);
				}
			}
		}
	}

}

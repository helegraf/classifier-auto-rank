package ranker.core.algorithms.decomposition.rankprediction;

import java.util.Arrays;
import java.util.List;

import ranker.core.algorithms.decomposition.regression.WEKARegressionRanker;
import weka.core.Instance;

public class RankRegressionRanker extends WEKARegressionRanker {

	public RankRegressionRanker(String name) {
		super(name);
	}

	@Override
	protected void modifyInstance(Instance instance, List<Integer> targetAttributes) {
		// replace target attributes by their ranks

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

package ranker.core.algorithms.preference;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.TreeMap;

import alors.Alors;
import alors.matrix_completion.cofirank.CofiConfig;
import alors.matrix_completion.cofirank.CofirankCPlusPlus;
import ranker.core.algorithms.Ranker;
import weka.core.Instance;
import weka.core.Instances;

public class ALORSRanker extends Ranker {

	private Alors alors;
	private CofiConfig coficonfig;

	public ALORSRanker(CofiConfig coficonfig) {
		this.coficonfig = coficonfig;
	}

	@Override
	public List<String> predictRankingforInstance(Instance instance) throws Exception {
		// convert features to ALORS
		double[] metafeatures = new double[this.data.numAttributes() - this.targetAttributes.size()];

		int index = 0;
		for (int i = 0; i < instance.numAttributes(); i++) {
			if (!this.targetAttributes.contains(i)) {
				metafeatures[index] = instance.value(i);
				index++;
			}
		}

		// make ALORS prediction and convert back to instances
		double[] predictedValues = alors.predictForFeatures(metafeatures);
		TreeMap<Double, String> predictionMapping = new TreeMap<>();
		for (int i = 0; i < predictedValues.length; i++) {
			predictionMapping.put(predictedValues[i], this.classifiersMap.get(this.targetAttributes.get(i)));
		}

		List<String> predictedOrdering = new ArrayList<>();
		predictionMapping.descendingMap().forEach((value, algorithm) -> predictedOrdering.add(algorithm));

		return predictedOrdering;
	}

	private double[][] getPortion(Instances data, boolean getClassifiers, List<Integer> targetAttributes) {
		int numAttributes = getClassifiers ? targetAttributes.size() : data.numAttributes() - targetAttributes.size();
		double[][] newData = new double[data.numInstances()][numAttributes];

		for (int i = 0; i < data.numInstances(); i++) {
			int index = 0;
			double[] instance = data.get(i).toDoubleArray();
			for (int j = 0; j < data.numAttributes(); j++) {
				if (targetAttributes.contains(j) == getClassifiers) {
					newData[i][index] = instance[j];
					index++;
				}
			}
		}

		return newData;
	}

	@Override
	public List<Double> getEstimates() {
		// TODO refactor so normal rankers don't have this method
		return null;
	}

	@Override
	protected void initialize() throws Exception {
		CofirankCPlusPlus cofirank = new CofirankCPlusPlus(coficonfig);
		alors = new Alors(cofirank);

		// convert the data to matrices: dataset feature portion; algorithm performance
		// portion
		double[][] trainPerformanceData = getPortion(this.data, true, this.targetAttributes);
		double[][] trainMetafeaturesData = getPortion(this.data, false, this.targetAttributes);

		System.out.println("init alorsranker with feature maps: M " + Arrays.deepToString(trainPerformanceData));
		System.out.println("U: " + Arrays.deepToString(trainMetafeaturesData));

		alors.completeMatrixAndPrepareColdStart(trainPerformanceData, trainMetafeaturesData);
	}

}

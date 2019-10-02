package datasets;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

public class ASLibScenario {

	// raw data
	private String scenarioName;
	private List<Algorithm> algorithms;
	private List<PerformanceMeasure> performanceMeasures;
	private double algorithmCutOffTime;
	private List<String> features;

	// data connections for easy access
	/**
	 * maps a repetition to a split index to instances belonging to this split index
	 */
	private Map<Integer, Map<Integer, List<Instance>>> splitMembers;
	/** maps an instance string id to */
	private Map<String, List<AlgorithmRun>> algorithmRunsOnInstance;

	public ASLibScenario(String scenarioName, List<PerformanceMeasure> performanceMeasures, List<Algorithm> algorithms,
			double algorithmCutOffTime, Map<String, List<AlgorithmRun>> algorithmRunsOnInstance,
			Map<Integer, Map<Integer, List<Instance>>> splitMembers, List<String> features) {
		super();
		this.scenarioName = scenarioName;
		this.performanceMeasures = performanceMeasures;
		this.algorithms = algorithms;
		this.algorithmCutOffTime = algorithmCutOffTime;
		this.algorithmRunsOnInstance = algorithmRunsOnInstance;
		this.splitMembers = splitMembers;
		this.features = features;
	}

	/**
	 * Get the testing data for a specified fold (includes the fold only). Assumes
	 * first repetition and first performance measure.
	 * 
	 * @param foldNum the fold to get
	 * @return the data as a {@link Instances} object
	 */
	public Instances getTestingDataForFold(int foldNum) {
		return this.getTestingDataForFold(foldNum, 1, performanceMeasures.get(0));
	}

	/**
	 * Get the testing data for a specified fold, repetition, and performance
	 * measure (includes the fold only).
	 * 
	 * @param foldNum       the fold to get
	 * @param repetitionNum the repetition for which the fold should be retrieved
	 * @param measure       the data for which measure for the algorithms should be
	 *                      included
	 * @return the data as a {@link Instances} object
	 */
	public Instances getTestingDataForFold(int foldNum, int repetitionNum, PerformanceMeasure measure) {
		// create instances
		ArrayList<Attribute> attInfo = new ArrayList<>();
		attInfo.add(new Attribute("id", true));
		int featureLen = splitMembers.get(1).get(1).get(0).getFeatures().length;
		for (int i = 0; i < featureLen; i++) {
			attInfo.add(new Attribute(features.get(i)));
		}
		for (int i = 0; i < algorithms.size(); i++) {
			attInfo.add(new Attribute("target:" + algorithms.get(i).getId()));
		}
		Instances testingData = new Instances(
				scenarioName + "_repetition-" + repetitionNum + "_fold-" + foldNum + "_test", attInfo,
				splitMembers.get(foldNum).size());

		// add instances to instances
		addInstancesOfFoldToDataForMeasure(foldNum, repetitionNum, measure, testingData);
		return testingData;
	}

	/**
	 * Get the training data for a specified fold (includes all folds except the
	 * given). Assumes first repetition and first performance measure.
	 * 
	 * @param foldNum the fold to get
	 * @return the data as a {@link Instances} object
	 */
	public Instances getTrainingDataForFold(int foldNum) {
		return this.getTrainingDataForFold(foldNum, 1, performanceMeasures.get(0));
	}

	/**
	 * Get the training data for a specified fold, repetition, and performance
	 * measure (includes all folds except the given).
	 * 
	 * @param foldNum       the fold to get
	 * @param repetitionNum the repetition for which the fold should be retrieved
	 * @param measure       the data for which measure for the algorithms should be
	 *                      included
	 * @return the data as a {@link Instances} object
	 */
	public Instances getTrainingDataForFold(int foldNum, int repetitionNum, PerformanceMeasure performanceMeasure) {
		// create instances
		ArrayList<Attribute> attInfo = new ArrayList<>();
		attInfo.add(new Attribute("id", true));
		int featureLen = splitMembers.get(1).get(1).get(0).getFeatures().length;
		for (int i = 0; i < featureLen; i++) {
			attInfo.add(new Attribute(features.get(i)));
		}
		for (int i = 0; i < algorithms.size(); i++) {
			attInfo.add(new Attribute("target:" + algorithms.get(i).getId()));
		}
		Instances trainingData = new Instances(
				scenarioName + "_repetition-" + repetitionNum + "_fold-" + foldNum + "_train", attInfo,
				splitMembers.get(foldNum).size());

		// add instances to instances
		for (int i = 0; i < splitMembers.get(repetitionNum).size(); i++) {
			if (i + 1 != foldNum) {
				addInstancesOfFoldToDataForMeasure(i + 1, repetitionNum, performanceMeasure, trainingData);
			}
		}

		return trainingData;
	}

	private void addInstancesOfFoldToDataForMeasure(int foldNum, int repetitionNum, PerformanceMeasure measure,
			Instances data) {
		splitMembers.get(repetitionNum).get(foldNum).forEach(instance -> {
			double[] attValues = new double[data.numAttributes()];
			for (int i = 0; i < instance.getFeatures().length; i++) {
				attValues[i + 1] = instance.getFeatures()[i];
			}

			for (int i = 0; i < algorithms.size(); i++) {
				// get run for the algorithm
				algorithmRunsOnInstance.computeIfAbsent(instance.getId(), k -> new ArrayList<>());
				List<AlgorithmRun> runs = algorithmRunsOnInstance.get(instance.getId());
				for (int j = 0; j < runs.size(); j++) {
					if (runs.get(j).getAlgorithm().getId().equals(algorithms.get(i).getId())) {
						attValues[i + instance.getFeatures().length + 1] = runs.get(j)
								.getAveragePerformanceMeasurement(measure);
						if (!measure.isMaximize()) {
							// invert measures that should be minimized
							attValues[i + instance.getFeatures().length
									+ 1] = -attValues[i + instance.getFeatures().length + 1];
						}
						break;
					}
				}
			}
			DenseInstance inst = new DenseInstance(1.0, attValues);
			data.add(inst);
			data.get(data.numInstances() - 1).setValue(0, instance.getId());

		});
	}

	public String getScenarioName() {
		return scenarioName;
	}

	public List<Algorithm> getAlgorithms() {
		return algorithms;
	}

	public List<PerformanceMeasure> getPerformanceMeasures() {
		return performanceMeasures;
	}

	public double getAlgorithmCutOffTime() {
		return algorithmCutOffTime;
	}

	public Map<Integer, Map<Integer, List<Instance>>> getSplitMembers() {
		return splitMembers;
	}

	public Map<String, List<AlgorithmRun>> getAlgorithmRunsOnInstance() {
		return algorithmRunsOnInstance;
	}

}

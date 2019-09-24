package datasets;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class AlgorithmRun {

	private Instance instance;
	private Algorithm algorithm;

	private Map<PerformanceMeasure, List<Double>> performanceMeasurements;
	private List<RunStatus> runStatus;

	private Map<PerformanceMeasure, Double> averagePerformanceMap;

	public AlgorithmRun(Instance instance, Algorithm algorithm, Map<PerformanceMeasure, List<Double>> performanceMeasurements, List<RunStatus> runStatus) {
		super();
		this.instance = instance;
		this.algorithm = algorithm;
		this.performanceMeasurements = performanceMeasurements;
		this.runStatus = runStatus;
		initialize();
	}

	public AlgorithmRun(Instance instance, Algorithm algorithm) {
		this(instance, algorithm, new HashMap<>(), new ArrayList<>());
	}

	private void initialize() {
		averagePerformanceMap = new HashMap<>();
		int numberOfValidMeasurements = 0;
		for (PerformanceMeasure measure : performanceMeasurements.keySet()) {
			double averagePerformance = 0;
			for (int i = 0; i < performanceMeasurements.get(measure).size(); i++) {
				double measurement = performanceMeasurements.get(measure).get(i);
				if (!Double.isNaN(measurement)) {
					averagePerformance += measurement;
					numberOfValidMeasurements++;
				}
			}
			averagePerformance /= numberOfValidMeasurements;
			averagePerformanceMap.put(measure, averagePerformance);
		}
	}

	public void addRun(PerformanceMeasure measure, double result, RunStatus runStatus) {
		if (!performanceMeasurements.containsKey(measure)) {
			performanceMeasurements.put(measure, new ArrayList<>());
		}
		performanceMeasurements.get(measure).add(result);
		this.runStatus.add(runStatus);
	}

	public Instance getInstance() {
		return instance;
	}

	public Algorithm getAlgorithm() {
		return algorithm;
	}

	public int getNumberOfRuns() {
		return runStatus.size();
	}

	public List<Double> getPerformanceMeasurements(PerformanceMeasure measure) {
		return performanceMeasurements.get(measure);
	}

	public double getAveragePerformanceMeasurement(PerformanceMeasure measure) {
		return averagePerformanceMap.get(measure);
	}

	public List<RunStatus> getRunStatus() {
		return runStatus;
	}

}

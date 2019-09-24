package datasets;

import java.util.List;
import java.util.Map;

public class ASLibScenario {

	private String scenarioName;

	private List<PerformanceMeasure> performanceMeasures;

	private List<Instance> instances;

	private List<Algorithm> algorithms;
	private double algorithmCutOffTime;

	private int amountOfSplits;
	private List<Split> splits;

	private List<AlgorithmRun> algorithmRuns;

	// help variables computed after all data is read in
	private Map<Instance, AlgorithmRun> instanceToRunMap;
	private Map<Split, Instance> splitToInstanceMap;

	public ASLibScenario(String scenarioName, List<PerformanceMeasure> performanceMeasures, List<Instance> instances, List<Algorithm> algorithms, double algorithmCutOffTime, int amountOfSplits, List<Split> splits,
			List<AlgorithmRun> algorithmRuns) {
		super();
		this.scenarioName = scenarioName;
		this.performanceMeasures = performanceMeasures;
		this.instances = instances;
		this.algorithms = algorithms;
		this.algorithmCutOffTime = algorithmCutOffTime;
		this.amountOfSplits = amountOfSplits;
		this.splits = splits;
		this.algorithmRuns = algorithmRuns;
	}

	public String getScenarioName() {
		return scenarioName;
	}

	public List<PerformanceMeasure> getPerformanceMeasures() {
		return performanceMeasures;
	}

	public List<Instance> getInstances() {
		return instances;
	}

	public List<Algorithm> getAlgorithms() {
		return algorithms;
	}

	public int getAmountOfSplits() {
		return amountOfSplits;
	}

	public List<Split> getSplits() {
		return splits;
	}

	public List<AlgorithmRun> getAlgorithmRuns() {
		return algorithmRuns;
	}

	public Map<Instance, AlgorithmRun> getInstanceToRunMap() {
		return instanceToRunMap;
	}

	public Map<Split, Instance> getSplitToInstanceMap() {
		return splitToInstanceMap;
	}

}

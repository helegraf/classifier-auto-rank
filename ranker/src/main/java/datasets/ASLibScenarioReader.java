package datasets;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import jaicore.basic.sets.SetUtil.Pair;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class ASLibScenarioReader {

	// helpers
	private List<Instance> instances;

	// for creating an aslib scenario
	private String scenarioName;
	private List<PerformanceMeasure> performanceMeasures;
	private List<Algorithm> algorithms;
	private List<String> features;
	private double algorithmCutOffTime;
	private Map<String, List<AlgorithmRun>> algorithmRunsOnInstance;
	/** maps repetition -> split -> datasets */
	private Map<Integer, Map<Integer, List<Instance>>> splitMembers;

	public ASLibScenario readASLibScenario(String pathToScenario) throws IOException {
		readDescriptionFile(pathToScenario);
		readInstanceFeatureFile(pathToScenario);
		readAlgorithmRunsFile(pathToScenario);
		readCVFile(pathToScenario);

		return new ASLibScenario(scenarioName, performanceMeasures, algorithms, algorithmCutOffTime,
				algorithmRunsOnInstance, splitMembers, features);
	}

	private void readCVFile(String pathToScenario) throws IOException {
		Instances data;
		try {
			DataSource source = new DataSource(pathToScenario + "/cv.arff");
			data = source.getDataSet();
		} catch (Exception exception) {
			throw new IOException("Cannot read cv.arff file.", exception);
		}

		splitMembers = new HashMap<>();
		for (int j = 0; j < data.numInstances(); j++) {
			weka.core.Instance instance = data.get(j);
			String instanceName = instance.stringValue(data.attribute("instance_id"));
			int repetition = (int) instance.value(data.attribute("repetition"));
			int fold = (int) instance.value(data.attribute("fold"));

			for (int i = 0; i < instances.size(); i++) {
				if (instances.get(i).getId().equals(instanceName)) {
					splitMembers.computeIfAbsent(repetition, k -> new HashMap<>());
					splitMembers.get(repetition).computeIfAbsent(fold, k -> new ArrayList<>());
					splitMembers.get(repetition).get(fold).add(instances.get(i));
				}
			}
		}
	}

	private void readAlgorithmRunsFile(String pathToScenario) throws IOException {
		algorithmRunsOnInstance = new HashMap<>();

		Instances data;
		try {
			DataSource source = new DataSource(pathToScenario + "/algorithm_runs.arff");
			data = source.getDataSet();
		} catch (Exception exception) {
			throw new IOException("Cannot read algorithm_runs.arff file.", exception);
		}
		for (weka.core.Instance instance : data) {
			String instanceName = instance.stringValue(data.attribute("instance_id"));
			int repetition = (int) instance.value(data.attribute("repetition"));
			String algorithmName = instance.stringValue(data.attribute("algorithm"));
			String runStatus = instance.stringValue(data.attribute("runstatus"));
			List<Pair<PerformanceMeasure, Double>> performanceMeasurements = new ArrayList<>();
			for (PerformanceMeasure measure : performanceMeasures) {
				double measurement = instance.value(data.attribute(measure.getName()));
				performanceMeasurements.add(new Pair<PerformanceMeasure, Double>(measure, measurement));
			}

			this.registerAlgorithmRun(instanceName, algorithmName, repetition, performanceMeasurements, runStatus);
		}
	}

	private void registerAlgorithmRun(String instanceName, String algorithmName, int repetition,
			List<Pair<PerformanceMeasure, Double>> performanceMeasurements, String runStatus) {
		algorithmRunsOnInstance.computeIfAbsent(instanceName, k -> new ArrayList<>());
		boolean isPresent = false;
		List<AlgorithmRun> runs = algorithmRunsOnInstance.get(instanceName);
		for (int i = 0; i < runs.size(); i++) {
			if (runs.get(i).getAlgorithm().getId().equals(algorithmName)) {
				isPresent = true;

				for (int j = 0; j < performanceMeasurements.size(); j++) {
					Pair<PerformanceMeasure, Double> pair = performanceMeasurements.get(j);
					runs.get(i).addRun(pair.getX(), pair.getY(), RunStatus.valueOf(runStatus.toUpperCase()));
				}

				break;
			}
		}

		if (!isPresent) {
			Instance instance = null;
			for (int i = 0; i < instances.size(); i++) {
				if (instances.get(i).getId().equals(instanceName)) {
					instance = instances.get(i);
					break;
				}
			}
			Algorithm algorithm = null;
			for (int i = 0; i < algorithms.size(); i++) {
				if (algorithms.get(i).getId().equals(algorithmName)) {
					algorithm = algorithms.get(i);
				}
			}

			AlgorithmRun run = new AlgorithmRun(instance, algorithm);
			for (int j = 0; j < performanceMeasurements.size(); j++) {
				Pair<PerformanceMeasure, Double> pair = performanceMeasurements.get(j);
				run.addRun(pair.getX(), pair.getY(), RunStatus.valueOf(runStatus.toUpperCase()));
			}

			algorithmRunsOnInstance.get(instanceName).add(run);
		}
	}

	private void readInstanceFeatureFile(String pathToScenario) throws IOException {
		Instances data;
		try {
			DataSource source = new DataSource(pathToScenario + "/feature_values.arff");
			data = source.getDataSet();
		} catch (Exception exception) {
			throw new IOException("Cannot read feature_values.arff file.", exception);
		}

		features = new ArrayList<>();
		for (int i = 2; i < data.numAttributes(); i++) {
			features.add(data.attribute(i).name());
		}

		HashMap<String, List<double[]>> featuresForInstance = new HashMap<>();
		for (weka.core.Instance instance : data) {
			String instanceName = instance.stringValue(0);
			double[] instanceFeatures = new double[instance.numAttributes() - 2];
			for (int i = 2; i < instance.numAttributes(); i++) {
				instanceFeatures[i - 2] = instance.value(i);
			}

			featuresForInstance.computeIfAbsent(instanceName, k -> new ArrayList<>());
			featuresForInstance.get(instanceName).add(instanceFeatures);
		}

		instances = new ArrayList<>();
		featuresForInstance.forEach((instanceName, instanceFeatures) -> {
			double[] avgFeatures = new double[instanceFeatures.get(0).length];
			for (int i = 0; i < instanceFeatures.size(); i++) {
				double[] instFeatures = instanceFeatures.get(i);
				for (int j = 0; j < instFeatures.length; j++) {
					avgFeatures[j] += instFeatures[j];
				}
			}

			for (int i = 0; i < avgFeatures.length; i++) {
				avgFeatures[i] /= instanceFeatures.size();
			}

			instances.add(new Instance(instanceName, avgFeatures));
		});

	}

	private void readDescriptionFile(String pathToScenario) throws IOException {
		// TODO if used extensively in the future, maybe replace with an actual yaml
		// reader
		performanceMeasures = new ArrayList<>();
		algorithms = new ArrayList<>();
		List<String> performanceMeasureNames = new ArrayList<>();
		List<PerformanceMeasureType> performanceMeasureTypes = new ArrayList<>();
		List<Boolean> performanceMeasureMaximizes = new ArrayList<>();

		List<String> lines = Files.readAllLines(Paths.get(pathToScenario, "description.txt"));
		for (int i = 0; i < lines.size(); i++) {
			String line = lines.get(i);

			if (line.startsWith("algorithm_cutoff_time:")) {
				algorithmCutOffTime = Double.parseDouble(line.replaceAll("algorithm_cutoff_time:", "").trim());
			} else if (line.startsWith("scenario_id:")) {
				scenarioName = line.replaceAll("scenario_id:", "").trim();
			} else if (line.startsWith("metainfo_algorithms:")) {
				i++;
				line = lines.get(i);
				while ((line.startsWith("\t") || line.startsWith("  ") || line.startsWith("    "))
						&& i < lines.size()) {
					String lineStart = line.startsWith("\t") ? "\t" : (line.startsWith("    ") ? "    " : "  ");
					lineStart = lineStart.concat(lineStart);
					String algorithmName = line.replaceAll(":", "").trim();
					boolean isAlgorithmDeterministic = false;

					i++;
					line = lines.get(i);
					while (line.startsWith(lineStart) && i < lines.size()) {
						if (line.trim().startsWith("deterministic:")) {
							String determinismIdentifier = line.replaceAll("deterministic:", "");
							isAlgorithmDeterministic = determinismIdentifier.contains("true");
						}

						i++;
						if (i < lines.size()) {
							line = lines.get(i);
						}
					}
					algorithms.add(new Algorithm(algorithmName, isAlgorithmDeterministic));
				}
				i--;
			} else if (line.startsWith("performance_measures:")) {
				if (line.replace("performance_measures", "").trim().endsWith(":")) {
					i++;
					line = lines.get(i);
					while (line.trim().startsWith("-") && i < lines.size()) {
						String performanceMeasureName = line.replaceAll("-", "").trim();
						performanceMeasureNames.add(performanceMeasureName);
						i++;
						line = lines.get(i);
					}
					i--;
				} else {
					String performanceMeasureName = line.replace("performance_measures:", "").trim();
					performanceMeasureNames.add(performanceMeasureName);
				}

			} else if (line.startsWith("maximize:")) {
				if (line.replace("maximize", "").trim().endsWith(":")) {
					i++;
					line = lines.get(i);
					while (line.trim().startsWith("-") && i < lines.size()) {
						String maximizationLine = line.replaceAll("-", "").trim();
						performanceMeasureMaximizes.add(maximizationLine.contains("true"));
						i++;
						line = lines.get(i);
					}
					i--;
				} else {
					String maximizationLine = line.replace("maximize:", "").trim();
					performanceMeasureMaximizes.add(maximizationLine.contains("true"));
				}
			} else if (line.startsWith("performance_type:")) {
				if (line.replace("performance_type", "").trim().endsWith(":")) {
					i++;
					line = lines.get(i);
					while (line.trim().startsWith("-") && i < lines.size()) {
						String performanceTypeLine = line.replaceAll("-", "").trim();
						PerformanceMeasureType performanceMeasureType = PerformanceMeasureType
								.valueOf(performanceTypeLine.toUpperCase());
						performanceMeasureTypes.add(performanceMeasureType);
						i++;
						line = lines.get(i);
					}
					i--;
				} else {
					String performanceTypeLine = line.replace("performance_type:", "").trim();
					PerformanceMeasureType performanceMeasureType = PerformanceMeasureType
							.valueOf(performanceTypeLine.toUpperCase());
					performanceMeasureTypes.add(performanceMeasureType);
				}
			}
		}

		for (int i = 0; i < performanceMeasureNames.size(); i++) {
			String performanceMeasureName = performanceMeasureNames.get(i);
			PerformanceMeasureType performanceMeasureType = performanceMeasureTypes.get(i);
			boolean maximize = performanceMeasureMaximizes.get(i);
			PerformanceMeasure performanceMeasure = new PerformanceMeasure(performanceMeasureName,
					performanceMeasureType, maximize);
			performanceMeasures.add(performanceMeasure);
		}
	}

}

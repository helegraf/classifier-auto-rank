//package datasets;
//
//import java.io.IOException;
//import java.nio.file.Files;
//import java.nio.file.Paths;
//import java.util.HashMap;
//import java.util.List;
//import java.util.Map;
//
//import jaicore.basic.sets.SetUtil.Pair;
//import weka.core.Instances;
//import weka.core.converters.ConverterUtils.DataSource;
//
//public class ASLibScenarioReader {
//
//	private String scenarioName;
//
//	private List<String> performanceMeasureNames;
//	private List<PerformanceMeasureType> performanceMeasureTypes;
//	private List<Boolean> performanceMeasureMaximizes;
//
//	private List<PerformanceMeasure> performanceMeasures;
//
//	private List<Algorithm> algorithms;
//	private double algorithmCutOffTime;
//
//	private Map<String, Map<String, AlgorithmRun>> algorithmToRunMap;
//
//	private List<Instance> instances;
//
//	public ASLibScenarioReader() {
//
//	}
//
//	public ASLibScenario readASLibScenario(String pathToScenario) {
//		readDescriptionFile(pathToScenario);
//		readInstanceFeatureFile(pathToScenario);
//		readAlgorithmRunsFile(pathToScenario);
//	}
//
//	public void readAlgorithmRunsFile(String pathToScenario) throws IOException {
//		algorithmToRunMap = new HashMap<>();
//
//		Instances instances;
//		try {
//			DataSource source = new DataSource(pathToScenario + "/algorithm_runs.arff");
//			instances = source.getDataSet();
//		} catch (Exception exception) {
//			throw new IOException("Cannot read algorithm_runs.arff file.", exception);
//		}
//		for (weka.core.Instance instance : instances) {
//			String instanceName = instance.stringValue(instances.attribute("instance_id"));
//			int repetition = (int) instance.value(instances.attribute("repitition"));
//			String algorithmName = instance.stringValue(instances.attribute("algorithm"));
//			Map<PerformanceMeasure, List<Double>> performanceMeasurements = new HashMap<>();
//			for (PerformanceMeasure measure : performanceMeasures) {
//				double measurement = instance.value(instances.attribute(measure.getName()));
//				performanceMeasurements.put(measure, value);
//			}
//		}
//	}
//
//	public void registerAlgorithmRun(String instanceName, String algorithmName, int repitition, List<Pair<PerformanceMeasure, Double>> performanceMeasures) {
//
//	}
//
//	public void readInstanceFeatureFile(String pathToScenario) throws IOException {
//		Instances instances;
//		try {
//			DataSource source = new DataSource(pathToScenario + "/feature_values.arff");
//			instances = source.getDataSet();
//		} catch (Exception exception) {
//			throw new IOException("Cannot read feature_values.arff file.", exception);
//		}
//		for (weka.core.Instance instance : instances) {
//			String instanceName = instance.stringValue(0);
//			double[] instanceFeatures = new double[instance.numAttributes() - 1];
//			for (int i = 1; i < instance.numAttributes(); i++) {
//				instanceFeatures[i - 1] = instance.value(i);
//			}
//			Instance aslibInstance = new Instance(instanceName, instanceFeatures);
//			this.instances.add(aslibInstance);
//		}
//	}
//
//	public void readDescriptionFile(String pathToScenario) throws IOException {
//		List<String> lines = Files.readAllLines(Paths.get(pathToScenario, "description.txt"));
//		for (int i = 0; i < lines.size(); i++) {
//			String line = lines.get(i);
//
//			if (line.startsWith("algorithm_cutoff_time:")) {
//				algorithmCutOffTime = Double.parseDouble(line.replaceAll("algorithm_cutoff_time:", "").trim());
//			}
//			if (line.startsWith("scenario_id:")) {
//				scenarioName = line.replaceAll("scenario_id:", "").trim();
//			}
//			if (line.startsWith("metainfo_algorithms:")) {
//				i++;
//				while (line.startsWith("\t") && i < lines.size()) {
//					line = lines.get(i);
//					String algorithmName = line.replaceAll(":", "").trim();
//					boolean isAlgorithmDeterministic = false;
//					i++;
//					while (line.startsWith("\t\t") && i < lines.size()) {
//						line = lines.get(i);
//						if (line.startsWith("deterministic:")) {
//							String determinismIdentifier = line.replaceAll("deterministic:", "");
//							isAlgorithmDeterministic = determinismIdentifier.contains("true");
//						}
//						i++;
//					}
//					algorithms.add(new Algorithm(algorithmName, isAlgorithmDeterministic));
//					i++;
//				}
//			}
//
//			if (line.startsWith("performance_measures:")) {
//				i++;
//				while (line.startsWith("-") && i < lines.size()) {
//					line = lines.get(i);
//					String performanceMeasureName = line.replaceAll("-", "").trim();
//					performanceMeasureNames.add(performanceMeasureName);
//					i++;
//				}
//			}
//			if (line.startsWith("maximize:")) {
//				i++;
//				while (line.startsWith("-") && i < lines.size()) {
//					line = lines.get(i);
//					String maximizationLine = line.replaceAll("-", "").trim();
//					performanceMeasureMaximizes.add(maximizationLine.contains("true"));
//					i++;
//				}
//			}
//
//			if (line.startsWith("performance_type:")) {
//				i++;
//				while (line.startsWith("-") && i < lines.size()) {
//					line = lines.get(i);
//					String performanceTypeLine = line.replaceAll("-", "").trim();
//					PerformanceMeasureType performanceMeasureType = PerformanceMeasureType.valueOf(performanceTypeLine.toUpperCase());
//					performanceMeasureTypes.add(performanceMeasureType);
//					i++;
//				}
//			}
//		}
//
//		for (int i = 0; i < performanceMeasureNames.size(); i++) {
//			String performanceMeasureName = performanceMeasureNames.get(i);
//			PerformanceMeasureType performanceMeasureType = performanceMeasureTypes.get(i);
//			boolean maximize = performanceMeasureMaximizes.get(i);
//			PerformanceMeasure performanceMeasure = new PerformanceMeasure(performanceMeasureName, performanceMeasureType, maximize);
//			performanceMeasures.add(performanceMeasure);
//		}
//	}
//
//}

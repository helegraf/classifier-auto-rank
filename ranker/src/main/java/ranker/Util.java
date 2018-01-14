package ranker;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import org.apache.commons.math3.stat.correlation.KendallsCorrelation;
import org.openml.apiconnector.io.OpenmlConnector;
import org.openml.apiconnector.xml.Data;
import org.openml.apiconnector.xml.Data.DataSet;
import org.openml.apiconnector.xml.DataFeature;
import org.openml.apiconnector.xml.DataFeature.Feature;

import de.upb.cs.is.jpl.api.algorithm.learningalgorithm.labelranking.instancebasedlabelranking.InstanceBasedLabelRankingLearningAlgorithm;
import de.upb.cs.is.jpl.api.algorithm.learningalgorithm.labelranking.instancebasedlabelranking.InstanceBasedLabelRankingLearningModel;
import de.upb.cs.is.jpl.api.algorithm.learningalgorithm.labelranking.labelrankingbypairwisecomparison.LabelRankingByPairwiseComparisonLearningAlgorithm;
import de.upb.cs.is.jpl.api.algorithm.learningalgorithm.labelranking.labelrankingbypairwisecomparison.LabelRankingByPairwiseComparisonLearningModel;
import de.upb.cs.is.jpl.api.dataset.defaultdataset.relative.Ranking;
import de.upb.cs.is.jpl.api.dataset.labelranking.LabelRankingDataset;
import de.upb.cs.is.jpl.api.exception.algorithm.PredictionFailedException;
import de.upb.cs.is.jpl.api.exception.algorithm.TrainModelsFailedException;
import rankerEvaluation.KendallRankCorrelation;
import rankerEvaluation.LeaveOneOut;
import rankerEvaluation.RankerEstimationProcedure;

import org.openml.apiconnector.xml.DataSetDescription;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.functions.GaussianProcesses;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SGD;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.SimpleLinearRegression;
import weka.classifiers.functions.SimpleLogistic;
import weka.classifiers.functions.VotedPerceptron;
import weka.classifiers.lazy.IBk;
import weka.classifiers.lazy.KStar;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.JRip;
import weka.classifiers.rules.M5Rules;
import weka.classifiers.rules.OneR;
import weka.classifiers.rules.PART;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.LMT;
import weka.classifiers.trees.M5P;
import weka.classifiers.trees.REPTree;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import wekaUtil.EstimationProcedure;
import wekaUtil.EvaluationMeasure;
import wekaUtil.PredictiveAccuary;
import wekaUtil.StratifiedMCCV;

/**
 * Helper class with various methods to facilitate data generation and analysis.
 * 
 * @author Helena Graf
 *
 */
public class Util {

	public static String apiKey;
	public static Charset charset = Charset.forName("UTF-8");
	public static Path apiKeyPath = FileSystems.getDefault().getPath("apikey.txt");
	public static Path jobsPath = FileSystems.getDefault().getPath("jobs.txt");
	public static Path dataSetIndexPath = FileSystems.getDefault().getPath("datasets.txt");
	public static Path resultsPath = FileSystems.getDefault().getPath("data");
	public static Path cacheDirectory = FileSystems.getDefault().getPath("data");
	public static Path logsDirectory = FileSystems.getDefault().getPath("logs");

	public static Classifier[] portfolio = { new BayesNet(), new NaiveBayes(), new NaiveBayesMultinomial(),
			new GaussianProcesses(), new LinearRegression(), new Logistic(), new MultilayerPerceptron(), new SGD(),
			new SMO(), new SimpleLinearRegression(), new SimpleLogistic(), new VotedPerceptron(), new IBk(),
			new KStar(), new DecisionTable(), new JRip(), new M5Rules(), new OneR(), new PART(), new ZeroR(),
			new DecisionStump(), new J48(), new LMT(), new M5P(), new RandomForest(), new RandomTree(), new REPTree() };

	public static String[] dataQualities;

	public static Map<Integer, double[]> dataQualitiesInstances = new HashMap<Integer, double[]>();
	public static Map<Classifier, Instances> classifierPerformances = new HashMap<Classifier, Instances>();
	public static Instances testSet;
	
	public static double testRanker (Ranker ranker, Instances instances) {
		// RegressionRanker ranker = new RegressionRanker();
		// ranker.buildRanker(instances);
		// ranker.predictRankingforInstance(instances.get(0)).forEach(action->System.out.println(action.getClass().getName()));
		RankerEstimationProcedure estim = new LeaveOneOut();
		double result = estim.estimate(ranker, new KendallRankCorrelation(), instances);
		return result;
	}

	public static void labelRankingTest() throws TrainModelsFailedException, PredictionFailedException {
		ArrayList<Integer> labels = new ArrayList<Integer>();
		labels.add(0);
		labels.add(1);
		ArrayList<double[]> features = new ArrayList<double[]>();
		double[] r1 = { 1, 0 };
		double[] r2 = { 0, 1 };
		double[] r3 = { 1, 1 };
		features.add(r1);
		features.add(r2);
		features.add(r3);
		ArrayList<Ranking> rankings = new ArrayList<Ranking>();
		int[] objectList = { 0, 1 };
		int[] compareOperators = { Ranking.COMPARABLE_ENCODING };
		Ranking ranking = new Ranking(objectList, compareOperators);
		rankings.add(ranking);
		int[] objectList1 = { 0, 1 };
		int[] compareOperators1 = { Ranking.COMPARABLE_ENCODING };
		Ranking ranking1 = new Ranking(objectList1, compareOperators1);
		rankings.add(ranking1);
		int[] objectList2 = { 0, 1 };
		int[] compareOperators2 = { Ranking.COMPARABLE_ENCODING };
		Ranking ranking2 = new Ranking(objectList2, compareOperators2);
		rankings.add(ranking2);
		LabelRankingDataset trainSet = new LabelRankingDataset(labels, features, rankings);

		InstanceBasedLabelRankingLearningAlgorithm algo = new InstanceBasedLabelRankingLearningAlgorithm();
		InstanceBasedLabelRankingLearningModel model = algo.train(trainSet);
		//LabelRankingByPairwiseComparisonLearningAlgorithm algo2 = new LabelRankingByPairwiseComparisonLearningAlgorithm();
		//LabelRankingByPairwiseComparisonLearningModel model2 = algo2.train(trainSet);

		ArrayList<Integer> labels1 = new ArrayList<Integer>();
		labels1.add(1);
		labels1.add(0);
		ArrayList<double[]> features1 = new ArrayList<double[]>();
		double[] r4 = { 0, 1 };
		features1.add(r4);
		ArrayList<Ranking> rankings1 = new ArrayList<Ranking>();
		rankings1.add(null);
		LabelRankingDataset testSet = new LabelRankingDataset(labels1, features1, rankings1);

		model.predict(testSet).forEach(item -> System.out.println(item));
		//model2.predict(testSet).forEach(item -> System.out.println(item));
	}
	
	public static Instances computeMetaFeatures(List<Integer> dids) throws Exception {
		// Prepare List of Attributes for Instances
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();

		// Qualities
		GlobalCharacterizer characterizer = new GlobalCharacterizer();
		String[] allDataQualities = characterizer.getIDs();
		Map<String, Integer> qualityIndices = new HashMap<String, Integer>();
		for (int i = 0; i < allDataQualities.length; i++) {
			String dataQuality = allDataQualities[i];
			qualityIndices.put(dataQuality, i);
			attributes.add(new Attribute(dataQuality));
		}

		// Classifiers
		Map<String, Integer> classifierIndices = new HashMap<String, Integer>();
		for (int i = 0; i < portfolio.length; i++) {
			String classifierName = portfolio[i].getClass().getName();
			classifierIndices.put(classifierName, allDataQualities.length + i);
			attributes.add(new Attribute(classifierName));
		}

		// Prepare instances
		Instances instances = new Instances("MetaDataOpenML_calculated", attributes, 0);
		HashMap<Integer, Instance> metaFeaturesForDataSets = new HashMap<Integer, Instance>();

		// Add meta features
		for (int dataSetId : dids) {
			Instance instance = new DenseInstance(attributes.size());
			Instances openMLData = null;
			try {
				openMLData = getInstancesById(dataSetId);
			} catch (IOException e) {
				// TODO logging maybe / outprint here
				System.out.println("Couldn't get " + dataSetId);
				continue;
			}
			
			Map<String, Double> dataQualities = characterizer.characterize(openMLData);
			dataQualities.forEach((dataQuality, value) -> {
				if (value != null) {
					instance.setValue(qualityIndices.get(dataQuality), value);
				}
			});
			metaFeaturesForDataSets.put(dataSetId, instance);
		}

		// Add performance results
		try (Stream<Path> paths = Files.walk(resultsPath)) {
			paths.filter(Files::isRegularFile).forEach(file -> {
				try {
					BufferedReader reader = Files.newBufferedReader(file, charset);
					String line = reader.readLine();
					if (line != null) {
						String fileName = file.getFileName().toString();
						fileName = fileName.substring(0, fileName.length() - 4);
						String[] parts = fileName.split("_");
						String classifierName = parts[0];
						int dataSetId = Integer.parseInt(parts[1]);
						if (dids.contains(dataSetId)) {
							Instance instance = metaFeaturesForDataSets.get(dataSetId);
							int attIndex = classifierIndices.get(classifierName);
							double value = Double.parseDouble(line);
							instance.setValue(attIndex, value);
						}
					}

				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			});
		}

		instances.addAll(metaFeaturesForDataSets.values());
		return instances;
	}

	public static Instances getMetaFeaturesFromOpenML() throws Exception {
		// Prepare List of Attributes for Instances
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();

		// Qualities
		OpenmlConnector client = new OpenmlConnector();
		String[] allDataQualities = client.dataQualitiesList().getQualities();
		Map<String, Integer> qualityIndices = new HashMap<String, Integer>();
		for (int i = 0; i < allDataQualities.length; i++) {
			String dataQuality = allDataQualities[i];
			qualityIndices.put(dataQuality, i);
			attributes.add(new Attribute(dataQuality));
		}

		// Classifiers
		Map<String, Integer> classifierIndices = new HashMap<String, Integer>();
		for (int i = 0; i < portfolio.length; i++) {
			String classifierName = portfolio[i].getClass().getName();
			classifierIndices.put(classifierName, allDataQualities.length + i);
			attributes.add(new Attribute(classifierName));
		}

		// Prepare instances
		Instances instances = new Instances("MetaDataOpenML", attributes, 0);
		HashMap<Integer, Instance> metaFeaturesForDataSets = new HashMap<Integer, Instance>();

		// Add meta features
		for (int dataSetId : getDataSetsFromIndex()) {
			Instance instance = new DenseInstance(attributes.size());
			Map<String, String> dataQualities = client.dataQualities(dataSetId).getQualitiesMap();
			dataQualities.forEach((dataQuality, value) -> {
				if (value != null) {
					instance.setValue(qualityIndices.get(dataQuality), Double.parseDouble(value));
				}
			});
			metaFeaturesForDataSets.put(dataSetId, instance);
		}

		// Add performance results
		try (Stream<Path> paths = Files.walk(resultsPath)) {
			paths.filter(Files::isRegularFile).forEach(file -> {
				try {
					BufferedReader reader = Files.newBufferedReader(file, charset);
					String line = reader.readLine();
					String fileName = file.getFileName().toString();
					fileName = fileName.substring(0, fileName.length() - 4);
					String[] parts = fileName.split("_");
					String classifierName = parts[0];
					int dataSetId = Integer.parseInt(parts[1]);
					Instance instance = metaFeaturesForDataSets.get(dataSetId);
					int attIndex = classifierIndices.get(classifierName);
					double value;
					if (line != null) {
						value = Double.parseDouble(line);
					} else {
						// TODO hack to avoid NaNs actually has to be worst value for performance measure used!
						value = 0;
					}
					instance.setValue(attIndex, value);

				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			});
		}

		instances.addAll(metaFeaturesForDataSets.values());
		return instances;
	}

	public static List<Integer> getDataSetsFromIndex() throws Exception {
		List<Integer> dataSets = new ArrayList<Integer>();
		BufferedReader reader = Files.newBufferedReader(dataSetIndexPath, charset);
		String line = null;
		while ((line = reader.readLine()) != null) {
			int dataSetId = Integer.parseInt(line);
			dataSets.add(dataSetId);
		}
		return dataSets;
	}

	public static void generatePerformanceMeasures(List<Classifier> classifiers, List<Instances> datasets,
			EvaluationMeasure evalM, EstimationProcedure estimProc, String filepath) {

		// Prepare table of results
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		for (int i = 0; i < datasets.size(); i++) {
			// Add index to attribute since they cannot have same names but some datasets do
			attributes.add(new Attribute(datasets.get(i).relationName() + " " + i));
		}
		Instances results = new Instances("PerformanceMeasures", attributes, 0);

		// Calculate performance of each classifier on each data set
		for (Classifier classifier : classifiers) {
			Instance instance = new DenseInstance(attributes.size());
			for (int i = 0; i < datasets.size(); i++) {
				try {
					double result = estimProc.estimate(classifier, evalM, datasets.get(i));
					instance.setValue(i, result);
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
			results.add(instance);
		}

		ArffSaver saver = new ArffSaver();
		saver.setInstances(results);
		try {
			saver.setFile(new File("./data/test.arff"));
			saver.writeBatch();
		} catch (IOException e) {
			e.printStackTrace();
		}

	}

	public static double generatePerformanceMeasure(Classifier classifier, Instances dataset) {
		// Get result of evaluation
		double result;
		try {
			result = new StratifiedMCCV(5, 0.3).estimate(classifier, new PredictiveAccuary(), dataset);
		} catch (Exception e) {
			System.err.println(
					classifier.getClass().getSimpleName() + " could not be evaluated on " + dataset.relationName());
			System.err.println(e.getMessage());
			result = 0;
		}
		return result;
	}

	public static Instances getInstancesById(int dataId) throws IOException {
		Instances dataset = null;

		// Get apiKey if not given
		if (apiKey == null) {
			BufferedReader reader = Files.newBufferedReader(apiKeyPath, charset);
			apiKey = reader.readLine();
		}

		// Get dataset from OpenML
		OpenmlConnector client = new OpenmlConnector();
		try {
			DataSetDescription description = client.dataGet(dataId);
			File file = description.getDataset(apiKey);
			// Instances convert
			DataSource source = new DataSource(file.getCanonicalPath());
			dataset = source.getDataSet();
			dataset.setClassIndex(dataset.numAttributes() - 1);
			Attribute targetAttribute = dataset.attribute(description.getDefault_target_attribute());
			dataset.setClassIndex(targetAttribute.index());
		} catch (Exception e) {
			// These are IOExceptions anyways in the extended sense of this method
			throw new IOException(e.getMessage());
		}
		return dataset;
	}

	public static void getDataFromOpenML(int numFeatures, int numInstances) throws Exception {
		// TODO more elaborate filters maybe? / be clear in passing options
		// For statistics
		int unfiltered;
		int filteredBNG = 0;
		int filteredARFF = 0;
		int filteredTarget = 0;
		int filteredNumeric = 0;
		int fitForAnalysis = 0;

		// For saving data sets
		BufferedWriter writer = Files.newBufferedWriter(FileSystems.getDefault().getPath("datasets_"+numFeatures+"_"+numInstances), charset);

		// OpenML connection
		OpenmlConnector client = new OpenmlConnector();

		// Get data sets that are active
		HashMap<String, String> map = new HashMap<String, String>();
		map.put("status", "active");
		Data data = client.dataList(map);
		DataSet[] data_raw = data.getData();
		unfiltered = data_raw.length;

		// Filter out data sets not fit for analysis
		for (int i = 0; i < data_raw.length; i++) {
			// Keep track of progress to see if something freezes
			System.out.println("Progress: " + (Math.round(i * 1.0 / data_raw.length * 100.0)));

			// No generated streaming data
			if (data_raw[i].getName().contains("BNG")) {
				filteredBNG++;
				continue;
			}

			// No non-.ARFF files
			if (!data_raw[i].getFormat().equals("ARFF")) {
				filteredARFF++;
				continue;
			}

			// Analyze features
			DataFeature dataFeature = client.dataFeatures(data_raw[i].getDid());
			Feature[] features = dataFeature.getFeatures();
			if (numFeatures>0 && features.length > numFeatures) {
				continue;
			}
			
			boolean noTarget = true;
			boolean numericTarget = true;
			for (int j = features.length - 1; j >= 0; j--) {
				if (features[j].getIs_target()) {
					noTarget = false;
					if (features[j].getDataType().equals("nominal")) {
						numericTarget = false;
					}
					break;
				}
			}
			
			// Analyze instances
			String numInst = data_raw[i].getQualityMap().get("NumberOfInstances");
			if (numInst==null) {
				System.out.println("Couldn't get num inst");
			} else {
				if (Double.parseDouble(numInst) > numInstances) {
					continue;
				}
			}

			// No non-existent target attributes
			if (noTarget) {
				filteredTarget++;
				continue;
			}

			// No numeric target attributes
			if (numericTarget) {
				filteredNumeric++;
				continue;
			}

			// Data is fit for analysis, save
			writer.write(Integer.toString(data_raw[i].getDid()));
			writer.newLine();
			fitForAnalysis++;

		}

		writer.close();

		// Print statistics
		System.out.println("Unfiltered: " + unfiltered);
		System.out.println("BNG: " + filteredBNG);
		System.out.println("ARFF: " + filteredARFF);
		System.out.println("No target: " + filteredTarget);
		System.out.println("Numeric target: " + filteredNumeric);
		System.out.println("Fit for analysis: " + fitForAnalysis);
	}
	

	public static void generateJobs() throws IOException {
		BufferedReader reader = Files.newBufferedReader(dataSetIndexPath, charset);
		BufferedWriter writer = Files.newBufferedWriter(jobsPath, charset);

		String line = null;
		while ((line = reader.readLine()) != null) {
			for (Classifier classifier : portfolio) {
				writer.write(classifier.getClass().getName() + "," + line);
				writer.newLine();
			}
		}
		reader.close();
		writer.close();
	}

	public static void performanceMeasures(String[] args) throws IOException, Exception {
		// Read input args
		if (args.length != 2) {
			throw new IllegalArgumentException("Wrong number of arguments supplied. Usage: Offset NumberOfJobs.");
		}
		int offset = Integer.parseInt(args[0]);
		int numberOfJobs = Integer.parseInt(args[1]);

		// Get jobs
		HashMap<Classifier, Integer> jobs = new HashMap<Classifier, Integer>();
		BufferedReader reader = Files.newBufferedReader(jobsPath, charset);
		String line = null;
		while (offset > 0 && (line = reader.readLine()) != null) {
			offset--;
		}
		while (numberOfJobs > 0 && (line = reader.readLine()) != null) {
			numberOfJobs--;
			String[] split = line.split(",");
			jobs.put(AbstractClassifier.forName(split[0], null), Integer.parseInt(split[1]));
		}
		reader.close();

		// Generate Performance Measures & save to files
		jobs.forEach((classifier, dataId) -> {
			try {
				Path dataPath = resultsPath.resolve(classifier.getClass().getName() + "_" + dataId + ".txt");
				BufferedWriter writer = Files.newBufferedWriter(dataPath, charset);
				Instances dataset = getInstancesById(dataId);
				double result = generatePerformanceMeasure(classifier, dataset);
				writer.write(Double.toString(result));
				writer.close();
			} catch (IOException e) {
				e.printStackTrace();
				System.err.println(e.getMessage());
			}
		});
	}
}

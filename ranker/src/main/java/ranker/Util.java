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
import org.openml.apiconnector.xml.DataSetDescription;
import org.openml.webapplication.fantail.dc.Characterizer;
import org.openml.webapplication.fantail.dc.statistical.Cardinality;
import org.openml.webapplication.features.GlobalMetafeatures;

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

	public static double calculateKendallRankCorrelation(Classifier[] predictedOrdering, Classifier[] actualOrdering) {
		// Convert Ordering given by Ranker into ranking
		double[] xArray = new double[portfolio.length];
		double[] yArray = new double[portfolio.length];

		for (int i = 0; i < portfolio.length; i++) {
			yArray[i] = i;
			for (int j = 0; j < portfolio.length; j++) {
				if (predictedOrdering[i].getClass().getName().equals(actualOrdering[j].getClass().getName())) {
					xArray[i] = j;
				}
			}
		}

		KendallsCorrelation correlation = new KendallsCorrelation();

		double result = correlation.correlation(xArray, yArray);
		return result;
	}

	public static void makeInstances(List<Integer> holdout) throws IOException {
		// aggregate for each classifier + add performance values
		// make new dataset
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		for (int i = 0; i < Util.dataQualities.length; i++) {
			attributes.add(new Attribute(Util.dataQualities[i]));
		}
		attributes.add(new Attribute("Performance"));
		Instances testset = new Instances("Testset", attributes, 0);
		for (Classifier classifier : portfolio) {
			Instances dataset = new Instances(classifier.getClass().getName(), attributes, 0);
			dataset.setClassIndex(attributes.size() - 1);
			// for each dataset : if result exists, add corresponding instance + performance
			// to dataset
			try (Stream<Path> paths = Files.walk(resultsPath)) {
				paths.filter(Files::isRegularFile)
						.filter(path -> path.getFileName().toString().contains(classifier.getClass().getName()))
						.forEach(path -> {
							try {
								BufferedReader reader = Files.newBufferedReader(path);
								String line = reader.readLine();
								if (line != null) {
									String element = path.getFileName().toString().split("_")[1];
									String toAdd = element.substring(0, element.length() - 4);
									int did = Integer.parseInt(toAdd);
									if (!holdout.contains(did)) {
										double result = Double.parseDouble(line);
										// copy instances
										double[] instanceValues = dataQualitiesInstances.get(did);
										instanceValues[instanceValues.length - 1] = result;
										Instance instance = new DenseInstance(dataQualities.length + 1, instanceValues);
										dataset.add(instance);
									}
								}
							} catch (IOException e) {
								// TODO Auto-generated catch block
								e.printStackTrace();
							}

						});
			}
			classifierPerformances.put(classifier, dataset);
			ArffSaver saver = new ArffSaver();
			saver.setInstances(dataset);
			try {
				saver.setFile(new File(
						resultsPath.resolve("Perf").resolve(classifier.getClass().getName() + ".arff").toString()));
				saver.writeBatch();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		for (int i : holdout) {
			double[] instanceValues = dataQualitiesInstances.get(i);
			Instance instance = new DenseInstance(dataQualities.length + 1, instanceValues);
			testset.add(instance);
		}
		// save test set
		ArffSaver saver = new ArffSaver();
		testset.setClassIndex(testset.numAttributes() - 1);
		saver.setInstances(testset);
		try {
			saver.setFile(new File(resultsPath.resolve("Perf").resolve("Test.arff").toString()));
			saver.writeBatch();
		} catch (IOException e) {
			e.printStackTrace();
		}
		testSet = testset;
	}

	public static void getAllDataQualities() throws Exception {
		// read all datasets from .txt
		BufferedReader reader = Files.newBufferedReader(dataSetIndexPath, charset);
		String line = null;
		while ((line = reader.readLine()) != null) {
			int did = Integer.parseInt(line);
			dataQualitiesInstances.put(did, getQualities(did));
		}
		// ad instance for each
	}

	public static double[] getQualities(int did) throws Exception {
		OpenmlConnector client = new OpenmlConnector();
		if (dataQualities == null) {
			dataQualities = client.dataQualitiesList().getQualities();
		}
		double[] instance = new double[dataQualities.length + 1];

		Map<String, String> qualities = client.dataQualities(did).getQualitiesMap();
		for (int i = 0; i < dataQualities.length; i++) {
			if (qualities.containsKey(dataQualities[i])) {
				if (!(qualities.get(dataQualities[i]) == null)) {
					instance[i] = Double.parseDouble(qualities.get(dataQualities[i]));
				} else {
					instance[i] = Double.NaN;
				}

			} else {
				instance[i] = Double.NaN;
			}
		}

		// calculate actual instance values
		// save in .ARFF for each

		GlobalMetafeatures allFeatures = new GlobalMetafeatures(null);
		List<Characterizer> characterizers = allFeatures.getCharacterizers();
		Cardinality cardinality = new Cardinality();
		characterizers.add(cardinality);

		Map<String, Double> results = new HashMap<String, Double>();
		Instances dataset = getInstancesById(did);
		for (Characterizer characterizer : characterizers) {
			results.putAll(characterizer.characterize(dataset));
		}

		return instance;

	}

	public static void generatePerformanceMeasures(List<Classifier> classifiers, List<Instances> datasets,
			EvaluationMeasure evalM, EstimationProcedure estimProc, String filepath) {
		// TODO refactor this method

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

		// return "String";
	}

	public static void mergePerformanceMeasures() throws IOException {

		// TODO decide how to parse sensibly
		// read all generated Files
		try (Stream<Path> paths = Files.walk(resultsPath)) {
			paths.filter(Files::isRegularFile).forEach(System.out::println);
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

	public static void getDataFromOpenML() throws Exception {
		// For statistics
		int unfiltered;
		int filteredBNG = 0;
		int filteredARFF = 0;
		int filteredTarget = 0;
		int filteredNumeric = 0;
		int fitForAnalysis = 0;

		// For saving data sets
		BufferedWriter writer = Files.newBufferedWriter(dataSetIndexPath, charset);

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
		System.out.println("Adds up: " + ((unfiltered
				- (filteredBNG + filteredARFF + filteredTarget + filteredNumeric + fitForAnalysis)) == 0));
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

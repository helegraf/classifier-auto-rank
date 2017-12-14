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

import org.openml.apiconnector.io.OpenmlConnector;
import org.openml.apiconnector.xml.Data;
import org.openml.apiconnector.xml.DataSetDescription;
import org.openml.apiconnector.xml.Data.DataSet;
import org.openml.apiconnector.xml.Data.DataSet.Quality;
import org.openml.apiconnector.xml.DataFeature;
import org.openml.apiconnector.xml.DataFeature.Feature;

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

/**
 * Helper class with various methods to facilitate data generation and analysis.
 * 
 * @author Helena Graf
 *
 */
public class Util {

	public static Classifier[] portfolio = { new BayesNet(), new NaiveBayes(), new NaiveBayesMultinomial(),
			new GaussianProcesses(), new LinearRegression(), new Logistic(), new MultilayerPerceptron(), new SGD(),
			new SMO(), new SimpleLinearRegression(), new SimpleLogistic(), new VotedPerceptron(), new IBk(),
			new KStar(), new DecisionTable(), new JRip(), new M5Rules(), new OneR(), new PART(), new ZeroR(),
			new DecisionStump(), new J48(), new LMT(), new M5P(), new RandomForest(), new RandomTree(), new REPTree() };

	public static void generatePerformanceMeasures(List<Classifier> classifiers, List<Instances> datasets,
			EvaluationMeasure evalM, EstimationProcedure estimProc, String filepath) {
//		// TODO implements so that it accepts Strings instead of data sets
//
//		// Prepare table of results
//		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
//		for (int i = 0; i < datasets.size(); i++) {
//			// Add index to attribute since they cannot have same names but some datasets do
//			// TODO better solution
//			attributes.add(new Attribute(datasets.get(i).relationName() + " " + i));
//		}
//		Instances results = new Instances("PerformanceMeasures", attributes, 0);
//
//		// Calculate performance of each classifier on each data set
//		for (Classifier classifier : classifiers) {
//			Instance instance = new DenseInstance(attributes.size());
//			for (int i = 0; i < datasets.size(); i++) {
//				try {
//					double result = estimProc.estimate(classifier, evalM, datasets.get(i));
//					instance.setValue(i, result);
//				} catch (Exception e) {
//					// TODO Auto-generated catch block
//					e.printStackTrace();
//				}
//			}
//			results.add(instance);
//		}
//
//		ArffSaver saver = new ArffSaver();
//		saver.setInstances(results);
//		try {
//			saver.setFile(new File("./data/test.arff"));
//			saver.writeBatch();
//		} catch (IOException e) {
//			// TODO Auto-generated catch block
//			e.printStackTrace();
//		}

		// return "String";
	}
	
	public static void getDatabyId(List<Integer> dataIds) {

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
		Path path = FileSystems.getDefault().getPath("src/main/ressources", "datasets.txt");
		Charset charset = Charset.forName("UTF-8");
		BufferedWriter writer = Files.newBufferedWriter(path, charset);

		// OpenML connection
		OpenmlConnector client = new OpenmlConnector();

		// Get data sets that are active
		HashMap<String, String> map = new HashMap<String, String>();
		map.put("status", "active");
		map.put("number_features", "20");
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
		Path jobsPath = FileSystems.getDefault().getPath("src/main/ressources", "jobs.txt");
		Path dataPath = FileSystems.getDefault().getPath("src/main/ressources", "datasets.txt");
		Charset charset = Charset.forName("UTF-8");
		BufferedReader reader = Files.newBufferedReader(dataPath,charset);
		BufferedWriter writer = Files.newBufferedWriter(jobsPath, charset);
		
		String line = null;
		while((line = reader.readLine()) != null) {
			for (Classifier classifier : portfolio) {
				writer.write(classifier.getClass().getName() + "," + line);
				writer.newLine();
			}
		}
		reader.close();
		writer.close();
	}
}

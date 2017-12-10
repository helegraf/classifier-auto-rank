package ranker;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;

import org.openml.apiconnector.io.OpenmlConnector;
import org.openml.apiconnector.xml.Data;
import org.openml.apiconnector.xml.DataSetDescription;
import org.openml.apiconnector.xml.Data.DataSet;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.functions.GaussianProcesses;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SGD;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.SMOreg;
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
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Main {

	public static void main(String[] args) throws Exception {
		// OpenML connection
		OpenmlConnector client = new OpenmlConnector();

		// Get some data
		HashMap<String, String> map = new HashMap<String, String>();
		map.put("status", "active");
		map.put("number_features", "20");
		Data data = client.dataList(map);
		DataSet[] data_raw = data.getData();
		ArrayList<String> datasets = new ArrayList();

		// TODO replace with meaningful logging
		System.out.println(data_raw.length);

		for (int i = 0; i < data_raw.length; i++) {
			// TODO filter data sets that contain BNG; only .ARFF files; only nominal target
			// features; only last feature = target
			DataSetDescription description = client.dataGet(data_raw[i].getDid());
			File file = description.getDataset("");
			datasets.add(file.getCanonicalPath());
			// Don't do this for now
			// try {
			// DataSource source = new DataSource(file.getCanonicalPath());
			// } catch (Exception e) {
			// e.printStackTrace();
			// }
			// TODO replace with meaningful logging
			System.out.println("Progress: " + (Math.round(i * 1.0 / data_raw.length * 100.0)));
		}

		// Initialize a list of all WEKA classifiers
		ArrayList<Classifier> classifiers = new ArrayList<Classifier>();
		// Bayes
		classifiers.add(new BayesNet());
		classifiers.add(new NaiveBayes());
		classifiers.add(new NaiveBayesMultinomial());
		// Functions
		classifiers.add(new GaussianProcesses());
		classifiers.add(new LinearRegression());
		classifiers.add(new Logistic());
		classifiers.add(new MultilayerPerceptron());
		classifiers.add(new SGD());
		classifiers.add(new SimpleLinearRegression());
		classifiers.add(new SimpleLogistic());
		classifiers.add(new SMO());
		classifiers.add(new SMOreg());
		classifiers.add(new VotedPerceptron());
		classifiers.add(new LibSVM());
		// Lazy
		classifiers.add(new IBk());
		classifiers.add(new KStar());
		// Rules
		classifiers.add(new DecisionTable());
		classifiers.add(new JRip());
		classifiers.add(new M5Rules());
		classifiers.add(new OneR());
		classifiers.add(new PART());
		classifiers.add(new ZeroR());
		// Trees
		classifiers.add(new DecisionStump());
		classifiers.add(new J48());
		classifiers.add(new LMT());
		classifiers.add(new M5P());
		classifiers.add(new RandomForest());
		classifiers.add(new RandomTree());
		classifiers.add(new REPTree());

		// Generate Performance Measures
		// Performance Measure Method has to be changed to accept string instead of
		// datasets
		// ranker.Util.generatePerformanceMeasures(classifiers, datasets,
		// predictiveAccuracy, estimproc);

		// TODO break big performance measures table down into many small tables with
		// attributes as metafeatures; last attribute (target) is performance

	}

}

package ranker;

import java.nio.charset.Charset;
import java.nio.file.FileSystems;
import java.nio.file.Path;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SGD;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.SimpleLogistic;
import weka.classifiers.functions.VotedPerceptron;
import weka.classifiers.lazy.IBk;
import weka.classifiers.lazy.KStar;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.JRip;
import weka.classifiers.rules.OneR;
import weka.classifiers.rules.PART;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.LMT;
import weka.classifiers.trees.REPTree;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;

/**
 * Helper class with various methods to facilitate data generation and analysis.
 * 
 * @author Helena Graf
 *
 */
public class Util {

	/**
	 * Set of used classifiers.
	 */
	public static final Classifier[] PORTFOLIO = { new BayesNet(), new NaiveBayes(), new NaiveBayesMultinomial(),
			new Logistic(), new MultilayerPerceptron(), new SGD(), new SMO(), new SimpleLogistic(),
			new VotedPerceptron(), new IBk(), new KStar(), new DecisionTable(), new JRip(), new OneR(), new PART(),
			new ZeroR(), new DecisionStump(), new J48(), new LMT(), new RandomForest(), new RandomTree(),
			new REPTree() };

	public static final String SYSTEM_SEPARATOR = FileSystems.getDefault().getSeparator();

	public static final Charset CHARSET = Charset.forName("UTF-8");

	public static Path dataSetIndexPath = FileSystems.getDefault().getPath("datasets_100_1000");
	public static Path resultsPath = FileSystems.getDefault().getPath("data");
	public static Path cacheDirectory = FileSystems.getDefault().getPath("data");
	public static final String RANKER_BUILD_TIMES = "RankerBuildTimes";
	public static final String RANKER_PREDICT_TIMES = "RankerPredictTimes";
	public static final String DATA_ID = "DataId";

	/**
	 * Used to separate values in generated .CSV files.
	 */
	public static final String CSV_SEPARATOR = ";";

	/**
	 * Used as a separator in the file names for the evaluation results of a
	 * classifier on a data set.
	 */
	public static final String CLASSIFIER_EVALUATION_RESULTS_SEPARATOR = "_";

	/**
	 * Name of the added column to the meta features containing the data set id (on
	 * OpenML).
	 */
	public static final String OPENML_DATASET_ID_FEATURE = "OpenML Data Set ID";

	/**
	 * The folder containing different .ARFF files of different sets of meta
	 * features computed for sets of data sets. Assumed to be located in the
	 * resources folder.
	 */
	public static final String META_DATA_FOLDER = "meta_data";

	/**
	 * The location of the file containing the full initial set of meta data for all
	 * initial data sets. The information has been gathered from OpenML:
	 */
	public static final String META_DATA_SMALL_DATA_SETS_OPENML = META_DATA_FOLDER + SYSTEM_SEPARATOR
			+ "metaData_allDataSets_OpenML.arff";

	/**
	 * The location of the file containing the full initial set of meta data for a
	 * selection of small data sets (containing no more than 100 features and 1000
	 * instances). The information has been computed.
	 */
	public static final String META_DATA_SMALL_DATA_SETS_COMPUTED = META_DATA_FOLDER + SYSTEM_SEPARATOR
			+ "metaData_smallDataSets_computed.arff";

	/**
	 * The location of the file containing the full initial set of meta data except
	 * for landmarkers for a selection of small data sets (containing no more than
	 * 100 features and 1000 instances).
	 */
	public static final String META_DATA_SMALL_DATA_SETS_COMPUTED_NO_PROBING = META_DATA_FOLDER + SYSTEM_SEPARATOR
			+ "metaData_smallDataSets_computed_noProbing.arff";

	/**
	 * The location of the file containing all initial landmarkers as meta data for
	 * a selection of small data sets (containing no more than 100 features and 1000
	 * instances).
	 */
	public static final String META_DATA_SMALL_DATA_SETS_COMPUTED_ONLY_PROBING = META_DATA_FOLDER + SYSTEM_SEPARATOR
			+ "metaData_smallDataSets_computed_onlyProbing.arff";

	/**
	 * The file enumerating jobs to be executed. Assumed to be located in the
	 * resources folder.
	 */
	public static final String JOBS_FILE = "jobs.txt";

	/**
	 * The file containing the OpenML API key. Assumed to be located in the
	 * resources folder. Has to be initialized to contain a valid OpenML API key
	 * before OpenML features can be used.
	 */
	public static final String APIKEY = "apikey.txt";

	/**
	 * The folder where the input and output of the program are stored.
	 */
	public static final String IO_FOLDER = "data";

	/**
	 * The folder where the results of the evaluation of classifiers on data sets
	 * are stored.
	 */
	public static final String CLASSIFIER_EVALUATION_RESULTS_FOLDER = IO_FOLDER + SYSTEM_SEPARATOR
			+ "classifier_evaluation_results";

	/**
	 * The folder where the indices of data sets (on OpenML) are stored.
	 */
	public static final String DATASET_INDICES_FOLDER = IO_FOLDER + SYSTEM_SEPARATOR + "dataset_indices";

	/**
	 * The file containing indices of data sets (on OpenML) of all initial data
	 * sets.
	 */
	public static final String DATASET_INDEX_ALL = DATASET_INDICES_FOLDER + SYSTEM_SEPARATOR + "alldatasets.txt";

	/**
	 * The file containing indices of data sets (on OpenML) of all initial small
	 * data sets (less than 100 features and 1000 instances).
	 */
	public static final String DATASET_INDEX_SMALL = DATASET_INDICES_FOLDER + SYSTEM_SEPARATOR
			+ "max_100_features_1000_instances_datasets.txt";

	/**
	 * The folder where some statistics gathered during the computation of meta
	 * features of data sets are stored.
	 */
	public static final String METAFEATURE_COMPUTATION_STATISTIC_FOLDER = IO_FOLDER + SYSTEM_SEPARATOR
			+ "metafeature_computation_statistics";

	/**
	 * The folder where the openML cache is located.
	 */
	public static final String OPENML_CACHE_FOLDER = IO_FOLDER + SYSTEM_SEPARATOR + "openML_cache";

}

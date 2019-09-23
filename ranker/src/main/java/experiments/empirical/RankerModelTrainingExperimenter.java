package experiments.empirical;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.concurrent.TimeUnit;

import org.aeonbits.owner.ConfigCache;
import org.aeonbits.owner.ConfigFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import de.upb.crc901.mlplan.core.AbstractMLPlanBuilder;
import de.upb.crc901.mlplan.core.MLPlan;
import de.upb.crc901.mlplan.core.MLPlanWekaRegressionBuilder;
import de.upb.crc901.mlplan.multiclass.wekamlplan.weka.model.MLPipeline;
import jaicore.basic.SQLAdapter;
import jaicore.basic.TimeOut;
import jaicore.concurrent.GlobalTimer;
import jaicore.experiments.ExperimentDBEntry;
import jaicore.experiments.ExperimentRunner;
import jaicore.experiments.IExperimentDatabaseHandle;
import jaicore.experiments.IExperimentIntermediateResultProcessor;
import jaicore.experiments.IExperimentSetEvaluator;
import jaicore.experiments.databasehandle.ExperimenterSQLHandle;
import jaicore.experiments.exceptions.ExperimentDBInteractionFailedException;
import jaicore.experiments.exceptions.ExperimentEvaluationFailedException;
import jaicore.experiments.exceptions.IllegalExperimentSetupException;
import jaicore.ml.WekaUtil;
import jaicore.ml.weka.dataset.splitter.ArbitrarySplitter;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffLoader.ArffReader;

public class RankerModelTrainingExperimenter implements IExperimentSetEvaluator {
	private static final String CLASSIFIER_FIELD = "classifier";
	private static final String PREPROCESSOR_FIELD = "preprocessor";

	private static final Logger L = LoggerFactory.getLogger(RankerModelTrainingExperimenter.class);

	private static RankerModelTrainingExperimenterConfig experimentConfig = ConfigCache.getOrCreate(RankerModelTrainingExperimenterConfig.class);
	private SQLAdapter adapter;
	private int experimentID;

	public RankerModelTrainingExperimenter(final File configFile) {
		super();
		Properties props = new Properties();
		try {
			props.load(new FileInputStream(configFile));
		} catch (IOException e) {
			L.error("Could not find or access config file {}", configFile, e);
			System.exit(1);
		}
		experimentConfig = ConfigFactory.create(RankerModelTrainingExperimenterConfig.class, props);
		if (experimentConfig.evaluationsTable() == null) {
			throw new IllegalArgumentException("No evaluations table (db.evalTable) set in config");
		}
	}

	@Override
	public void evaluate(final ExperimentDBEntry experimentEntry, final IExperimentIntermediateResultProcessor processor) throws ExperimentEvaluationFailedException {
		try {
			experimentID = experimentEntry.getId();
			Map<String, String> experimentValues = experimentEntry.getExperiment().getValuesOfKeyFields();

			// 1. get the whole dataset for the classifier (data handler, all 800 datasets) - maybe do this offline so guarenteed
			BufferedReader reader = new BufferedReader(new FileReader("resources/" + experimentValues.get("targetclassifier") + ".arff"));
			ArffReader arff = new ArffReader(reader);
			Instances data = arff.getData();
			data.setClassIndex(data.numAttributes()-1);
			
			// 2. do a seeded 70/30 split
			long seed = Long.parseLong(experimentValues.get("seed"));
			L.info("Split instances with seed {}", seed);
			List<Instances> arbitrarySplit = new ArbitrarySplitter().split(data, seed, .7);
			
			// 3. train mlplan
			/* initialize ML-Plan with the same config file that has been used to specify the experiments */
			MLPlanWekaRegressionBuilder builder = AbstractMLPlanBuilder.forWekaRegression();
			builder.withWEKARegressionConfiguration();
			builder.withTimeOut(new TimeOut(Integer.parseInt(experimentValues.get("timeout")), TimeUnit.SECONDS));
			builder.withNodeEvaluationTimeOut(new TimeOut(Integer.parseInt(experimentValues.get("evaluationTimeout")), TimeUnit.SECONDS));
			builder.withCandidateEvaluationTimeOut(new TimeOut(Integer.parseInt(experimentValues.get("evaluationTimeout")), TimeUnit.SECONDS));
			builder.withNumCpus(experimentEntry.getExperiment().getNumCPUs());

			MLPlan mlplan = new MLPlan(builder, arbitrarySplit.get(0));
			mlplan.setLoggerName("mlplan");
			mlplan.setTimeout(new Integer(experimentValues.get("timeout")), TimeUnit.SECONDS);
			mlplan.setRandomSeed(new Integer(experimentValues.get("seed")));

			L.info("Build mlplan classifier");
			Classifier optimizedClassifier = mlplan.call();

			L.info("Open timeout tasks: {}", GlobalTimer.getInstance().getActiveTasks());
			
			// 4. save model (models/split_i/classifier.txt) (for which classifier it is! -> label name)
			String classifierName = experimentValues.get("targetclassifier");
			SerializationHelper.write("models/split_" + experimentValues.get("seed") + "/"+ classifierName + ".txt", mlplan.getSelectedClassifier());
			
			// 5. test model
			Evaluation eval = new Evaluation(arbitrarySplit.get(0));
			L.info("Assess test performance...");
			eval.evaluateModel(optimizedClassifier, arbitrarySplit.get(1));
			
			L.info("Test error was {}. Internally estimated error for this model was {}", eval.rootMeanSquaredError(), mlplan.getInternalValidationErrorOfSelectedClassifier());
			Map<String, Object> results = new HashMap<>();
			results.put("rmse", eval.rootMeanSquaredError());
			if (mlplan.getSelectedClassifier() instanceof MLPipeline) {
				results.put(CLASSIFIER_FIELD, WekaUtil.getClassifierDescriptor(((MLPipeline) mlplan.getSelectedClassifier()).getBaseClassifier()));
				results.put(PREPROCESSOR_FIELD, ((MLPipeline) mlplan.getSelectedClassifier()).getPreprocessors().toString());
			} else {
				results.put(CLASSIFIER_FIELD, WekaUtil.getClassifierDescriptor(mlplan.getSelectedClassifier()));
			}
			
			processor.processResults(results);
			L.info("Experiment done.");
		}
		catch (Exception e) {
			throw new ExperimentEvaluationFailedException(e);
		}
	}
	
	public static void main (String [] args) throws ExperimentDBInteractionFailedException, IllegalExperimentSetupException {
		File configFile = new File("conf/mlplan-ranker-models.properties");
		IExperimentDatabaseHandle dbHandle = new ExperimenterSQLHandle(experimentConfig);
		ExperimentRunner runner = new ExperimentRunner(experimentConfig,new RankerModelTrainingExperimenter(configFile),dbHandle);
		runner.randomlyConductExperiments(1, false);
	}
}

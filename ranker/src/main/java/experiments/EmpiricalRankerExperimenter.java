package experiments;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;

import org.aeonbits.owner.ConfigCache;
import org.aeonbits.owner.ConfigFactory;
import org.apache.log4j.Level;
import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.eventbus.Subscribe;

import de.upb.crc901.mlplan.core.events.ClassifierFoundEvent;
import de.upb.crc901.mlplan.multiclass.wekamlplan.weka.model.MLPipeline;
import jaicore.basic.SQLAdapter;
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
import ranker.core.algorithms.Ranker;
import ranker.core.algorithms.decomposition.regression.MLPlanRegressionRanker;
import ranker.core.algorithms.decomposition.regression.WEKARegressionRanker;
import ranker.core.evaluation.EvaluationHelper;
import ranker.core.evaluation.strategies.MCCV;
import ranker.core.evaluation.strategies.NFoldCrossvalidation;
import ranker.core.evaluation.strategies.NFoldCrossvalidationOnlyOneFold;
import ranker.core.evaluation.strategies.RankerEstimationProcedure;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;

public class EmpiricalRankerExperimenter implements IExperimentSetEvaluator {
	private static final String CLASSIFIER_FIELD = "classifier";
	private static final String PREPROCESSOR_FIELD = "preprocessor";
	private int experimentID;

	private static final org.slf4j.Logger L = LoggerFactory.getLogger(EmpiricalRankerExperimenter.class);

	private static EmpiricalRankerExperimenterConfig experimentConfig = ConfigCache
			.getOrCreate(EmpiricalRankerExperimenterConfig.class);
	private SQLAdapter adapter;

	public EmpiricalRankerExperimenter(final File configFile) {
		super();
		Properties props = new Properties();
		try {
			props.load(new FileInputStream(configFile));
		} catch (IOException e) {
			L.error("Could not find or access config file {}", configFile, e);
			System.exit(1);
		}
		experimentConfig = ConfigFactory.create(EmpiricalRankerExperimenterConfig.class, props);
		if (experimentConfig.evaluationsTable() == null) {
			throw new IllegalArgumentException("No evaluations table (db.evalTable) set in config");
		}
	}

	@Override
	public void evaluate(final ExperimentDBEntry experimentEntry,
			final IExperimentIntermediateResultProcessor processor) throws ExperimentEvaluationFailedException {
		try {
			List<Logger> loggers = Collections.<Logger>list(LogManager.getCurrentLoggers());
			loggers.add(LogManager.getRootLogger());
			for ( Logger logger : loggers ) {
				System.out.println("WARN level logger " + logger.getName());
			    logger.setLevel(Level.WARN);
		//	    logger.removeAllAppenders();
	//		    logger.addAppender(new NullAppender());
			}
//			Logger.getRootLogger().removeAllAppenders();
//			Logger.getRootLogger().addAppender(new NullAppender());
			
			
			this.experimentID = experimentEntry.getId();
			this.adapter = new SQLAdapter(experimentConfig.getDBHost(), experimentConfig.getDBUsername(),
					experimentConfig.getDBPassword(), experimentConfig.getDBDatabaseName());
			Map<String, String> experimentValues = experimentEntry.getExperiment().getValuesOfKeyFields();

			L.info("Evaluate Ranker {}", experimentValues.get("algorithm"));
			// Load the correct data
			BufferedReader reader = new BufferedReader(new FileReader("resources/" + experimentValues.get("metafeatures") + "_" + experimentValues.get("imputation_variant")  + ".arff"));
			ArffReader arff = new ArffReader(reader);
			Instances data = arff.getData();
			List<Integer> targetAttributes = detectTargetAttributes(data);
			for (int i = 0; i < data.numInstances(); i++) {
				data.get(i).setWeight(1);
			}
			
			// Init ranker
			Ranker ranker = getRanker(experimentValues.get("algorithm"), experimentValues, experimentEntry);
					
			// Evalate
			RankerEstimationProcedure estim = getEstimationProcedure(experimentValues.get("split"), experimentValues.get("seed"), experimentValues);
			
			List<Double> result;
			//if (!(ranker instanceof PreferenceRanker)) {
				result = EvaluationHelper.evaluateRegressionRanker(estim, ranker, data, targetAttributes);
			//} else {
				//result = EvaluationHelper.evaluateRanker(new MCCV(5,.7),ranker, data, targetAttributes);
			//}
			L.info("result: {}", result);	
			L.info("num results: {}",result.size());
			
			Map<String, Object> results = new HashMap<>();
			// kendall, kendall_tied, rmse, loss, b3l
			results.put("kendall", result.get(0));
			results.put("kendall_tied", result.get(1));
			results.put("max_diff", result.get(2));
			int index = 2;
//			if (!(ranker instanceof PreferenceRanker)) {
				results.put("rmse", result.get(3));
				index++;
//			} else {
//				results.put("rmse", -1);
//			}
			
			for (int i = 1; i < 23; i++) {
				results.put("bl_" + i, result.get(index + i));
			}
			
			index += 23;
			
			results.put("ndcg_at_3",result.get(index));
			results.put("ndcg_at_5",result.get(index+1));
			results.put("ndcg_at_10",result.get(index+2));
			results.put("ndcg_at_22",result.get(index+3));
			
			if (ranker instanceof MLPlanRegressionRanker) {
				results.put(CLASSIFIER_FIELD, ((MLPlanRegressionRanker)ranker).getSelectedModelString());
			} else if (ranker instanceof WEKARegressionRanker) {
				results.put(CLASSIFIER_FIELD, ((WEKARegressionRanker)ranker).getAlgorithm());
			} else {
				results.put(CLASSIFIER_FIELD, ranker.getClass().getSimpleName());
			}
			
			processor.processResults(results);
			L.info("Experiment done.");
		} catch (Exception e) {
			throw new ExperimentEvaluationFailedException(e);
		}
	}
	
	private RankerEstimationProcedure getEstimationProcedure(String procedure, String seed, Map<String, String> experimentValues) {

		if (procedure.endsWith("Folds")) {
			if (experimentValues.get("fold") != null) {
				return new NFoldCrossvalidationOnlyOneFold(Integer.parseInt(procedure.split("_")[0]),Integer.parseInt(experimentValues.get("fold")), seed);
			} else {
				return new NFoldCrossvalidation(Integer.parseInt(procedure.split("_")[0]), seed);
			}
		} else if (procedure.endsWith("MCCV")) {
			int numbers = Integer.parseInt(procedure.split("_")[0]);
			double portions = Double.parseDouble(procedure.split("_")[1]);
			return new MCCV(numbers, portions, seed);
		}
	
		throw new IllegalArgumentException(procedure + " is not a valid estimation procedure.");
	}

	private List<Integer> detectTargetAttributes(Instances data) {
		ArrayList<Integer> result = new ArrayList<>();
		
		for(int i = 0; i < data.numAttributes(); i++) {
			if (data.attribute(i).name().startsWith("weka.")) {
				result.add(i);
			}
		}
		
		return result;
	}

	private Ranker getRanker(String algorithm, Map<String, String> experimentValues, ExperimentDBEntry experimentEntry) {
		if (algorithm.startsWith("mlplan")) {
			return getMLPLanRanker(algorithm, experimentValues, experimentEntry);
		} else if (algorithm.startsWith("weka")) {
			return getWEKARanker(algorithm);
		}
//		} else if (algorithm.startsWith("preference")) {
//			return getPreferenceRanker(algorithm);
//		} else if (algorithm.startsWith("baseline")) {
//			return getBaselineRanker(algorithm);
//		}
		
		throw new IllegalArgumentException(algorithm + " is not a valid ranker.");
	}

//	private Ranker getBaselineRanker(String algorithm) {
//		switch(algorithm) {
//		case "baseline_bestAlgorithm" : return new BestAlgorithmRanker();
//		default : throw new IllegalArgumentException(algorithm + " is not a valid baseline ranker.");
//		}
//	}
//
//	private Ranker getPreferenceRanker(String algorithm) {
//		switch(algorithm) {
//		case "preference_PCR" : return new PairwiseComparisonRanker();
//		case "preference_IBLR": return new InstanceBasedLabelRankingRanker();
//		case "preference_IBLR_KY" : return new InstanceBasedLabelRankingKemenyYoung();
//		case "preference_IBLRKYS_SQRTN" : return new InstanceBasedLabelRankingKemenyYoungSQRTN();
//		default : throw new IllegalArgumentException(algorithm + " is not a valid preference ranker.");
//		}
//	}

	private Ranker getWEKARanker(String algorithm) {
		return new WEKARegressionRanker(algorithm);
	}

	private Ranker getMLPLanRanker(String algorithm, Map<String, String> experimentValues, ExperimentDBEntry experimentEntry) {
		int divisor = experimentValues.get("fold") != null ? 1 : getRepetitionTimes(experimentValues.get("split"));
		return new MLPlanRegressionRanker(Integer.parseInt(experimentValues.get("seed")),
				experimentEntry.getExperiment().getNumCPUs(), Integer.parseInt(experimentValues.get("timeout"))/divisor,
				Integer.parseInt(experimentValues.get("evaluationTimeout")),this, algorithm.split("_")[1]);
	}

	private int getRepetitionTimes(String string) {
		return Integer.parseInt(string.split("_")[0]);
	}

	@Subscribe
	public void rcvHASCOSolutionEvent(final ClassifierFoundEvent e) {
		if (adapter != null) {
			try {
				String classifier = "";
				String preprocessor = "";
				if (e.getSolutionCandidate() instanceof MLPipeline) {
					MLPipeline solution = (MLPipeline) e.getSolutionCandidate();
					preprocessor = solution.getPreprocessors().isEmpty() ? ""
							: solution.getPreprocessors().get(0).toString();
					classifier = WekaUtil.getClassifierDescriptor(solution.getBaseClassifier());
				} else {
					classifier = WekaUtil.getClassifierDescriptor(e.getSolutionCandidate());
				}
				Map<String, Object> eval = new HashMap<>();
				eval.put("experiment_id", experimentID);
				eval.put(PREPROCESSOR_FIELD, preprocessor);
				eval.put(CLASSIFIER_FIELD, classifier);
				if (!Double.isNaN(e.getScore())) {
					eval.put("rmse", e.getScore());
				} else {
					L.warn("Uploading incomplete intermediate solution!");
				}

				eval.put("time_train", e.getTimestamp());
				adapter.insert(experimentConfig.evaluationsTable(), eval);
			} catch (Exception e1) {
				L.error("Could not store hasco solution in database", e1);
			}
		} else {
			L.error("no adapter!");
		}
	}

	public static void main(String[] args)
			throws ExperimentDBInteractionFailedException, IllegalExperimentSetupException {
		File configFile = new File("conf/empirical_ranker.properties");
		IExperimentDatabaseHandle dbHandle = new ExperimenterSQLHandle(experimentConfig);
		ExperimentRunner runner = new ExperimentRunner(experimentConfig, new EmpiricalRankerExperimenter(configFile),
				dbHandle);
		runner.randomlyConductExperiments(1, false);
	}
}

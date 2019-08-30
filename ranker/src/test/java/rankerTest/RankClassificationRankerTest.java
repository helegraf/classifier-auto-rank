package rankerTest;

import java.io.BufferedReader;
import java.io.FileReader;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import org.junit.Test;

import ranker.core.algorithms.decomposition.conflictresolution.DefaultStrategy;
import ranker.core.algorithms.decomposition.conflictresolution.MeanRankStrategy;
import ranker.core.algorithms.decomposition.rankprediction.RankClassificationRanker;
import ranker.core.evaluation.EvaluationHelper;
import ranker.core.evaluation.strategies.MCCV;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;

public class RankClassificationRankerTest {

	@Test
	public void makePredictionsWithDefaultStrategy() throws Exception {
		// import arff
		BufferedReader reader = new BufferedReader(
				new FileReader(Paths.get("src", "test", "resources", "noProbing_nonan_noid.arff").toString()));
		ArffReader arff = new ArffReader(reader);
		Instances data = arff.getData();
//		Instances train = new Instances(data, 0, data.numInstances() - 10);
//		Instances test = new Instances(data, data.numInstances() - 10, 10);

		RankClassificationRanker ranker = new RankClassificationRanker(RandomForest.class.getName());
		ranker.setConflictResolutionStrategy(new DefaultStrategy());

		List<Double> measures = EvaluationHelper.evaluateRanker(new MCCV(3, .7, "RankClassificationRankerTEST"), ranker,
				data, getTargetAttributes(data));
		
		System.out.println(measures);
	}

	private List<Integer> getTargetAttributes(Instances data) {
		List<Integer> targetAttributes = new ArrayList<>();

		for (int i = 0; i < data.numAttributes(); i++) {
			if (data.attribute(i).name().startsWith("weka")) {
				targetAttributes.add(i);
			}
		}

		return targetAttributes;
	}
	
	@Test
	public void makePredictionsWithMeanRankStrategy() throws Exception {
		// import arff
		BufferedReader reader = new BufferedReader(
				new FileReader(Paths.get("src", "test", "resources", "noProbing_nonan_noid.arff").toString()));
		ArffReader arff = new ArffReader(reader);
		Instances data = arff.getData();
//		Instances train = new Instances(data, 0, data.numInstances() - 10);
//		Instances test = new Instances(data, data.numInstances() - 10, 10);

		RankClassificationRanker ranker = new RankClassificationRanker(RandomForest.class.getName());
		ranker.setConflictResolutionStrategy(new MeanRankStrategy());

		List<Double> measures = EvaluationHelper.evaluateRanker(new MCCV(3, .7, "RankClassificationRankerTEST"), ranker,
				data, getTargetAttributes(data));
		
		System.out.println(measures);
	}
}

package rankerTest;

import java.io.BufferedReader;
import java.io.FileReader;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import ranker.core.algorithms.Ranker;
import ranker.core.algorithms.decomposition.conflictresolution.HighestProbabilityStrategy;
import ranker.core.algorithms.decomposition.conflictresolution.ExpectedRankStrategy;
import ranker.core.algorithms.decomposition.rankprediction.RankClassificationRanker;
import ranker.core.algorithms.decomposition.rankprediction.RankRegressionRanker;
import ranker.core.algorithms.decomposition.regression.WEKARegressionRanker;
import ranker.core.algorithms.preference.InstanceBasedLabelRankingKemenyYoung;
import ranker.core.algorithms.preference.InstanceBasedLabelRankingKemenyYoungSQRTN;
import ranker.core.algorithms.preference.InstanceBasedLabelRankingRanker;
import ranker.core.algorithms.preference.PairwiseComparisonRanker;
import ranker.core.algorithms.preference.PairwiseComparisonWEKARanker;
import ranker.core.evaluation.EvaluationHelper;
import ranker.core.evaluation.strategies.MCCV;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;

@RunWith(Parameterized.class)
public class RankerTest {

	private Ranker ranker;

	public RankerTest(Ranker ranker) {
		this.ranker = ranker;
	}

	@Parameters
	public static Collection<Ranker> data() {
		RankClassificationRanker rankClassificationRankerProbabilityStrategy = new RankClassificationRanker(
				RandomForest.class.getName());
		rankClassificationRankerProbabilityStrategy.setConflictResolutionStrategy(new HighestProbabilityStrategy());
		RankClassificationRanker rankClassificationRankerMeanStrategy = new RankClassificationRanker(
				RandomForest.class.getName());
		rankClassificationRankerMeanStrategy.setConflictResolutionStrategy(new ExpectedRankStrategy());
		return Arrays.asList(rankClassificationRankerProbabilityStrategy, rankClassificationRankerMeanStrategy,
				new RankRegressionRanker(RandomForest.class.getName()),
				new WEKARegressionRanker(RandomForest.class.getName()), new InstanceBasedLabelRankingRanker(),
				new InstanceBasedLabelRankingKemenyYoung(), new InstanceBasedLabelRankingKemenyYoungSQRTN(),
				new PairwiseComparisonRanker(), new PairwiseComparisonWEKARanker());
				//new MLPlanRegressionRanker(0, 4, 330, 15, null, "rfClean"));
	}

	@Test
	public void testMakePredictions() throws Exception {
		// import arff
		BufferedReader reader = new BufferedReader(
				new FileReader(Paths.get("src", "test", "resources", "noProbing_nonan_noid.arff").toString()));
		ArffReader arff = new ArffReader(reader);
		Instances data = arff.getData();

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
}

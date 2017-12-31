package ranker;

import java.util.ArrayList;
import java.util.List;

import org.openml.apiconnector.io.OpenmlConnector;
import org.openml.webapplication.fantail.dc.Characterizer;
import org.openml.webapplication.fantail.dc.statistical.Cardinality;
import org.openml.webapplication.features.GlobalMetafeatures;

import de.upb.cs.is.jpl.api.algorithm.learningalgorithm.labelranking.instancebasedlabelranking.InstanceBasedLabelRankingLearningAlgorithm;
import de.upb.cs.is.jpl.api.algorithm.learningalgorithm.labelranking.instancebasedlabelranking.InstanceBasedLabelRankingLearningModel;
import de.upb.cs.is.jpl.api.dataset.defaultdataset.relative.Ranking;
import de.upb.cs.is.jpl.api.dataset.labelranking.LabelRankingDataset;

public class Main {

	public static void main(String[] args) throws Exception {
		GlobalMetafeatures features = new GlobalMetafeatures(null);
		List<Characterizer> chars = features.getCharacterizers();
		chars.add(new Cardinality());
		chars.add(new DefaultAccuracy());
		List<String> qual = new ArrayList<String>();
		for (Characterizer ch : chars) {
			String [] cra = ch.getIDs();
			for (String str : cra) {
				qual.add(str);
			}
		}
		System.out.println(qual.size());
		
		OpenmlConnector client = new OpenmlConnector();
		String []  realQual = client.dataQualitiesList().getQualities();
		System.out.println(realQual.length);
		
		for (String str : realQual) {
			if (qual.contains(str)) {
				System.out.println("Contains " + str);
			} else {
				System.out.println("Does not contain " + str);
			}
		}
		
		
//		Util.getAllDataQualities();
//		ArrayList<Integer> holdout = new ArrayList<Integer>();
//		holdout.add(7);
//		Util.makeInstances(holdout);
//		Ranker ranker = new Ranker(RankingMode.REGRESSION);
//		ranker.buildRanker(Util.classifierPerformances);
//		List<Classifier> ranking = ranker.rank(Util.testSet.firstInstance());
//		ranking.forEach(classifier->System.out.println(classifier.getClass().getName()));
		
//		ArrayList<Integer> labels = new ArrayList<Integer>();
//		labels.add(0);
//		labels.add(1);
//		ArrayList<double[]> features = new ArrayList<double[]>();
//		double [] r1 = {1,0};
//		double [] r2 = {0,1};
//		double [] r3 = {1,1};
//		features.add(r1);
//		features.add(r2);
//		features.add(r3);
//		ArrayList<Ranking> rankings = new ArrayList<Ranking>();
//		int [] objectList = {0,1};
//		int [] compareOperators = {Ranking.COMPARABLE_ENCODING};
//		Ranking ranking = new Ranking(objectList, compareOperators);
//		rankings.add(ranking);
//		int [] objectList1 = {0,1};
//		int [] compareOperators1 = {Ranking.COMPARABLE_ENCODING};
//		Ranking ranking1 = new Ranking(objectList1, compareOperators1);
//		rankings.add(ranking1);
//		int [] objectList2 = {1,0};
//		int [] compareOperators2 = {Ranking.COMPARABLE_ENCODING};
//		Ranking ranking2 = new Ranking(objectList2, compareOperators2);
//		rankings.add(ranking2);
//		LabelRankingDataset test = new LabelRankingDataset(labels, features, rankings);
//		InstanceBasedLabelRankingLearningAlgorithm algo = new InstanceBasedLabelRankingLearningAlgorithm();
//		
//		InstanceBasedLabelRankingLearningModel model = algo.train(test);
//		
//		ArrayList<Integer> labels1 = new ArrayList<Integer>();
//		labels1.add(1);
//		labels1.add(0);
//		ArrayList<double[]> features1 = new ArrayList<double[]>();
//		double [] r4 = {0,0};
//		features1.add(r4);
//		ArrayList<Ranking> rankings2 = new ArrayList<Ranking>();
//		rankings2.add(null);
//		LabelRankingDataset train = new LabelRankingDataset(labels1, features1, rankings2);
//		
//		model.predict(train).forEach(item->System.out.println(item));
		
	}

}

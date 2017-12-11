package ranker;

import java.util.List;
import java.util.Map;

import weka.classifiers.Classifier;

public class Ranker {

	private RankingMode rankingMode = RankingMode.REGRESSION;
	private Map<Classifier,String> classifierPerformances;
	private List<Classifier> regressionAlgorithms;
	private Object preferenceLearner;

	public Ranker() {
		// TODO Initialize with pre-computed values: all WEKA classifiers + openML data sets
	}
	
	public Ranker(Map<Classifier,String> classifierPerformances) {
		this.classifierPerformances = classifierPerformances;
	}
	
	public void buildRanker () {
		// TODO Train regression / preference algorithm
	}
	
	public List<Classifier> rank (String dataset) {
		// TODO Return ranking
		return null;
	}
	
	public RankingMode getRankingMode() {
		return rankingMode;
	}


	public void setRankingMode(RankingMode rankingMode) {
		this.rankingMode = rankingMode;
	}


}

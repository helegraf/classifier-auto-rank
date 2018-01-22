package rankerEvaluation;

import java.util.List;

import ranker.algorithms.PerfectRanker;
import ranker.algorithms.Ranker;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

public class Loss implements RankerEvaluationMeasure {

	@Override
	public double evaluate(Ranker ranker, Instances train, Instances test, List<Integer> targetAttributes) {
		// TODO Auto-generated method stub
		try {
			Ranker oracle = new PerfectRanker();
			oracle.buildRanker(train, targetAttributes);
			ranker.buildRanker(train, targetAttributes);
			
			for (Instance instance : test) {
				List<Classifier> perfectRanking = oracle.predictRankingforInstance(instance);
				List<Classifier> predictedRanking = ranker.predictRankingforInstance(instance);
				
				for (int i = 0; i < perfectRanking.size(); i++) {
					if (perfectRanking.get(0).getClass().getName().equals(predictedRanking.get(i).getClass().getName())) {
						
						break;
					}
				}
			}
			
		} catch (Exception e) {
			// TODO log
		}
		return 0;
	}

}

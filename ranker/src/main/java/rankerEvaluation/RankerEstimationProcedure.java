package rankerEvaluation;

import java.util.List;

import ranker.algorithms.Ranker;
import weka.core.Instances;

public interface RankerEstimationProcedure {
	public double estimate (Ranker ranker, RankerEvaluationMeasure evaluatioProcedure, Instances instances, List<Integer> targetAttributes);
}

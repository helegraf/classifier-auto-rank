package rankerEvaluation;

import ranker.Ranker;
import weka.core.Instances;

public interface RankerEstimationProcedure {
	public double estimate (Ranker ranker, RankerEvaluationMeasure evaluatioProcedure, Instances instances);
}

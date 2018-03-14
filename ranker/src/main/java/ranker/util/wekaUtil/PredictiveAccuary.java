package ranker.util.wekaUtil;

import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;

/**
 * Computes the predictive accuracy of a classifier. 
 * 
 * @author Helena Graf
 *
 */
public class PredictiveAccuary implements EvaluationMeasure {

	@Override
	public double evaluate(Classifier classifier, Instances train, Instances test) throws Exception {
		Evaluation evaluation = new Evaluation(train);
		classifier.buildClassifier(train);
		evaluation.evaluateModel(classifier, test);
		double result = evaluation.pctCorrect();
		return result;
	}

}

package rankerEvaluation;

import java.util.ArrayList;

import ranker.algorithms.Ranker;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class LeaveOneOut implements RankerEstimationProcedure {

	@Override
	public double estimate(Ranker ranker, RankerEvaluationMeasure evaluationProcedure, Instances instances) {
		
		double result = 0;
		for (int i = 0; i < instances.numInstances(); i++) {
			Instances train = new Instances(instances);
			Instance remove = train.remove(i);
			ArrayList<Attribute> attributes = new ArrayList<Attribute>();
			for (int attribute = 0; attribute < instances.numAttributes(); attribute++) {
				attributes.add(instances.attribute(attribute));
			}
			Instances test = new Instances("Test", attributes, 0);
			test.add(remove);

			result += evaluationProcedure.evaluate(ranker, train, test);
		}
		
		result /= instances.numInstances();
		return result;
	}

}

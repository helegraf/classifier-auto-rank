package experiments;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

import ranker.core.algorithms.Ranker;
import ranker.core.algorithms.decomposition.regression.SavedModelRanker;
import ranker.core.evaluation.EvaluationHelper;
import ranker.core.evaluation.strategies.MCCV;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;

public class SavedModelRankerEvaluator {
	
	public static void main(String[] args) throws Exception {
		for (int i = 0; i < 5; i++) {
			// Load data
			BufferedReader reader = new BufferedReader(new FileReader("resources/complete.arff"));
			ArffReader arff = new ArffReader(reader);
			Instances data = arff.getData();		
			
			// Init ranker
			Ranker ranker = new SavedModelRanker(i);
			List<Integer> targetAttributes = new ArrayList<>();
			for (int j = 104; j < 126; j++) {
				targetAttributes.add(j);
			}
			
			List<Double> result = EvaluationHelper.evaluateRanker(new MCCV(5,.7,"noSeed"),ranker, data, targetAttributes);
			System.out.println(result);			
		}
	}
}

package ranker.core.evaluation;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import ranker.Util;
import ranker.core.algorithms.Ranker;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Divides data into n different splits of n-1 instances as a train set and the
 * remaining instance as the test set. Averages the result of these n runs,
 * leaving out NaN values (these are still included in
 * {@link #getDetailedEvaluationResults()} thought).
 * 
 * @author Helena Graf
 *
 */
public class LeaveOneOut extends RankerEstimationProcedure {

	private static final String CSV_SEPERATOR ="; ";
	
	@Override
	public List<Double> estimate(Ranker ranker, List<RankerEvaluationMeasure> measures, Instances data,
			List<Integer> targetAttributes) throws Exception {
		// Initialize variables
		for (RankerEvaluationMeasure measure : measures) {
			detailedEvaluationResults.put(measure.getClass().getSimpleName(), new ArrayList<Double>());
		}
		detailedEvaluationResults.put(Util.DATA_ID, new ArrayList<>());
		detailedEvaluationResults.put(Util.RANKER_BUILD_TIMES, new ArrayList<>());
		
		// Evaluate all instances separately
		for (int i = 0; i < data.numInstances(); i++) {
			Instances train = new Instances(data);
			Instance instanceToTest = train.remove(i);
			ArrayList<Attribute> attributes = new ArrayList<Attribute>();
			for (int j = 0; j < data.numAttributes(); j++) {
				attributes.add(data.attribute(j));
			}
			Instances test = new Instances("Test", attributes, 0);
			test.add(instanceToTest);
			detailedEvaluationResults.get(Util.DATA_ID).add((double) instanceToTest.value(0));
			//remove DataIds, we do not want them as values for testing
			train.deleteAttributeAt(0);
			test.deleteAttributeAt(0);
			List<Integer> decrementedTargetAttributes = new ArrayList<>();
			targetAttributes.forEach(number -> decrementedTargetAttributes.add(new Integer(number-1)));
			evaluateChunk(ranker, train, test, measures, decrementedTargetAttributes);
		}
		writeCSVFile(ranker, data);

		// Average results
		int numInstancesCalculated = data.numInstances();
		List<Double> results = new ArrayList<Double>();
		for (String measure : detailedEvaluationResults.keySet()) {
			double result = 0;
			for (double value : detailedEvaluationResults.get(measure)) {
				if (!Double.isNaN(value)) {
					result += value;
				} else {
					numInstancesCalculated--;
				}
			}
			result /= numInstancesCalculated;
			results.add(result);
		}

		return results;
	}
	
	private void writeCSVFile(Ranker ranker, Instances instance) {

	
		 try {
			 FileWriter fw = new FileWriter(new File(ranker.getClass().getSimpleName()+ "_" + instance.relationName() + ".csv"));
			 BufferedWriter writer = new BufferedWriter(fw);

			List<String> keys= createListOfKeys(new HashSet<String> (detailedEvaluationResults.keySet()));
			writer.write(createCSVHeader(keys));
			int numberOfResults = detailedEvaluationResults.get(Util.DATA_ID) == null ? 0: detailedEvaluationResults.get(Util.DATA_ID).size();
			System.out.println("number of Results is: " + numberOfResults);
			for(int i = 0; i< numberOfResults; i++) {
				writer.newLine();
				String lineToWrite = createDataLine(i, keys);
				writer.write(lineToWrite);
			}
			writer.flush();
			writer.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	private String createDataLine(int dataset, List<String> keys) {
		String toReturn = "";
		for(String key: keys) {
			toReturn += detailedEvaluationResults.get(key).get(dataset);
			toReturn += CSV_SEPERATOR;
		}
		return toReturn.trim().substring(0, toReturn.length()-2);
	}

	private String createCSVHeader(List<String> input ) {
		String toReturn = "";
		for(String headerName :input) {
			toReturn += headerName + CSV_SEPERATOR; 
		}
		//remove last whitespace and komma
		return toReturn.trim().substring(0, toReturn.length()-2);
		
	}

	private List<String> createListOfKeys(Set<String> keySet) {
		List <String> toReturn = new ArrayList<>();
		if(keySet.contains(Util.DATA_ID) && keySet.contains(Util.RANKER_BUILD_TIMES)) {
			toReturn.add(Util.DATA_ID);
			toReturn.add(Util.RANKER_BUILD_TIMES);
			keySet.remove(Util.DATA_ID);
			keySet.remove(Util.RANKER_BUILD_TIMES);
		}else {
			throw new RuntimeException("DataID or Ranker build times not in keyset of detailed evaluation results");
		}
		keySet.forEach(entry -> toReturn.add(entry));
		return toReturn;
	}
}

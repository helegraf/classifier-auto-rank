package ranker.util.wekaUtil;

import java.util.ArrayList;
import java.util.List;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

public class WekaHelper {
	
	/**
	 * Replaces missing values in the Instances object given with the
	 * {@link weka.filters.unsupervised.attribute.ReplaceMissingValues} filter.
	 * Returns new Instances object without missing values.
	 * 
	 * @param data
	 * @return
	 * @throws Exception
	 */
	public static Instances replaceMissingValues(Instances data) throws Exception {
		// Ensure no missing vaules
		// TODO extract method
		ReplaceMissingValues filter = new ReplaceMissingValues();
		filter.setInputFormat(data);
		for (int i = 0; i < data.numInstances(); i++) {
			filter.input(data.instance(i));
		}
		filter.batchFinished();
		Instances newData = filter.getOutputFormat();
		Instance processed;
		while ((processed = filter.output()) != null) {
			newData.add(processed);
		}
		return newData;
	}
	
	public static Instances subSet(Instances instances, int fromIndex, int toIndex) {
		List<Instance> test = instances.subList(fromIndex, toIndex);
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		for (int attribute = 0; attribute < instances.numAttributes(); attribute++) {
			attributes.add(instances.attribute(attribute));
		}
		Instances testInst = new Instances(instances.relationName(), attributes, 0);
		test.forEach(instance -> testInst.add(instance));
		return testInst;
	}
}

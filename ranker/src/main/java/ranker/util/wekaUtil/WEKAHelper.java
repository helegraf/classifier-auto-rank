package ranker.util.wekaUtil;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

/**
 * A collection of recurring tasks regarding WEKA.
 * 
 * @author Helena Graf
 *
 */
public class WEKAHelper {

	/**
	 * Replaces missing values in the Instances object with the
	 * {@link weka.filters.unsupervised.attribute.ReplaceMissingValues} filter.
	 * Returns new Instances object without missing values.
	 * 
	 * @param instances
	 *            Instances with missing values
	 * @return New Instances without missing values
	 * @throws Exception
	 *             If
	 *             {@link weka.filters.unsupervised.attribute.ReplaceMissingValues#setInputFormat(Instances)}
	 *             causes an Exception
	 */
	public static Instances replaceMissingValues(Instances instances) throws Exception {
		// Initialize filter
		ReplaceMissingValues filter = new ReplaceMissingValues();
		filter.setInputFormat(instances);

		// Process all instances
		for (int i = 0; i < instances.numInstances(); i++) {
			filter.input(instances.instance(i));
		}
		filter.batchFinished();

		// Create new Instances object
		Instances filteredInstances = filter.getOutputFormat();
		Instance processed;
		while ((processed = filter.output()) != null) {
			filteredInstances.add(processed);
		}

		return filteredInstances;
	}

	/**
	 * Creates a new Instances object containing only the instances within and
	 * including the specified bounds. The new Instances is created as a shallow
	 * copy, it is only a view of the instances in the original object.
	 * 
	 * @param instances
	 *            The Instances from which the subset is created
	 * @param fromIndex
	 *            The lower bound
	 * @param toIndex
	 *            The upper bound
	 * @return A subset of the given Instances
	 */
	public static Instances subSet(Instances instances, int fromIndex, int toIndex) {
		List<Instance> subList = instances.subList(fromIndex, toIndex);
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		for (int attribute = 0; attribute < instances.numAttributes(); attribute++) {
			attributes.add(instances.attribute(attribute));
		}
		Instances subSet = new Instances(instances.relationName(), attributes, instances.numInstances());
		subList.forEach(instance -> subSet.add(instance));
		return subSet;
	}

	/**
	 * Takes an Instances object and saves it in the .arff format with the filename
	 * as the relation name.
	 * 
	 * @param instances
	 * @throws IOException
	 */
	public static void saveAsArff(Instances instances) throws IOException {
		ArffSaver saver = new ArffSaver();
		saver.setInstances(instances);
		saver.setFile(new File(instances.relationName()));
		saver.writeBatch();
	}
}

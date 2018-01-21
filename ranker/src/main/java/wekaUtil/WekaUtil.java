package wekaUtil;

import weka.core.Instance;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

public class WekaUtil {
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
		while  ((processed=filter.output())!=null) {
			newData.add(processed);
		}
		return newData;
	}
}

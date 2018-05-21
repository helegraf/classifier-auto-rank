package ranker.util;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ranker.Util;
import ranker.core.algorithms.Ranker;
import weka.core.Instances;

/**
 * Class that converts evaluation results to .csv files and writes them on the
 * disk.
 * 
 * @author Helena Graf
 *
 */
public class CSVHelper {

	/**
	 * Converts the given ranker and data set into a path. To be used as a
	 * destination for .csv files containing evaluation results of the given ranker
	 * and meta data set.
	 * 
	 * @param ranker
	 *            The ranker
	 * @param instances
	 *            The meta data set that was used in the evaluation in the ranker
	 * @return The converted path
	 */
	public static String getCSVPath(Ranker ranker, Instances instances) {
		return Util.RANKER_EVALUATION_RESULTS + ranker.getClass().getSimpleName() + "_" + instances.relationName()
				+ ".csv";
	}

	/**
	 * Writes the given contents to the specified destination. It is assumed that
	 * the contents map columns titles to the contents of that column, and that all
	 * columns contain equally many values. The order of the column contents is
	 * preserved.
	 * 
	 * @param destination
	 *            The destination (including the file name) the results are written
	 *            to
	 * @param contents
	 *            The contents to be written
	 * @throws IOException
	 *             If an Exception occurs while writing to the disk
	 */
	public static void writeCSVFile(String destination, Map<String, List<Double>> contents) throws IOException {
		// initialize writer
		FileWriter fileWriter = new FileWriter(new File(destination.toString()));
		BufferedWriter writer = new BufferedWriter(fileWriter);

		// create the header
		List<String> keys = createListOfColumnTitles(new HashSet<String>(contents.keySet()));
		writer.write(createCSVHeader(keys));

		// write each line
		for (int i = 0; i < contents.size(); i++) {
			writer.newLine();
			writer.write(createCSVDataLine(i, keys, contents));
		}

		// close writer
		writer.flush();
		writer.close();
	}

	private static List<String> createListOfColumnTitles(Set<String> columnTitles) {
		// initialize list to return
		List<String> toReturn = new ArrayList<>();
		
		// ensure that data id is in the first column
		Set<String> columnTitlesCopy = new HashSet<String>(columnTitles);
		if (columnTitlesCopy.contains(Util.DATA_ID) && columnTitlesCopy.contains(Util.RANKER_BUILD_TIMES)) {
			//TODO there must be a better way for this -> don't assume always exist / data id is first when deleting for analysis etc. ?
			toReturn.add(Util.DATA_ID);
			toReturn.add(Util.RANKER_BUILD_TIMES);
			columnTitlesCopy.remove(Util.DATA_ID);
			columnTitlesCopy.remove(Util.RANKER_BUILD_TIMES);
		} else {
			throw new RuntimeException("DataID or Ranker build times not in keyset of detailed evaluation results");
		}
		// add column titles to list and return
		columnTitlesCopy.forEach(entry -> toReturn.add(entry));
		return toReturn;
	}

	private static String createCSVHeader(List<String> columnTitles) {
		// append column titles in the correct oder
		String toReturn = "";
		for (String columnTitle : columnTitles) {
			toReturn += columnTitle + Util.CSV_SEPARATOR;
		}

		// remove last whitespace and semicolon
		return toReturn.trim().substring(0, toReturn.length() - 2);

	}

	private static String createCSVDataLine(int index, List<String> columnTitles, Map<String, List<Double>> contents) {
		// append values in the correct order
		String toReturn = "";
		for (String key : columnTitles) {
			toReturn += contents.get(key).get(index);
			toReturn += Util.CSV_SEPARATOR;
		}

		// remove last whitespace and semicolon
		return toReturn.trim().substring(0, toReturn.length() - 2);
	}

}

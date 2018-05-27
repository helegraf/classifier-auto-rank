package ranker.util;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ranker.Util;
import ranker.core.algorithms.Ranker;
import weka.core.Instances;

/**
 * Class that helps with the reading and writing of .csv files.
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
	 *            The meta data set that was used in the evaluation of the ranker
	 * @return The converted path
	 */
	public static String getCSVPath(Ranker ranker, Instances instances) {
		return getCSVPath(ranker.getClass().getSimpleName(), instances.relationName());
	}

	/**
	 * Converts the given ranker name and data set name into a path. TO be used as a
	 * destination for .csv files containing evaluation results of the given ranker
	 * and meta data set.
	 *  
	 * @param rankerName The ranker name
	 * @param instancesName The meta data set name that was used in the evaluation of the ranker.
	 * @return The converted Path
	 */
	public static String getCSVPath(String rankerName, String instancesName) {
		return Util.RANKER_EVALUATION_RESULTS + Util.SYSTEM_SEPARATOR + rankerName + Util.RANKER_EVALUATION_RESULTS_SEPARATOR + instancesName + ".csv";
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

	/**
	 * Finds the index of a given column in a header line of a .csv file. If the
	 * name occurs more than one time, the index of the first occurrence is
	 * returned.
	 * 
	 * @param line
	 *            The header line
	 * @param column
	 *            The name of the column
	 * @return The column index
	 * @throws ColumnNotFoundException
	 *             If the given name does not occur in the line
	 */
	public static int findIndexOfColumn(String line, String column) throws ColumnNotFoundException {
		// Find first occurrence of index
		int columnIndex = -1;
		String[] columns = line.split(Util.CSV_SEPARATOR);
		for (int i = 0; i < columns.length; i++) {
			if (columns[i].equals(column)) {
				columnIndex = i;
				break;
			}
		}

		// Throw exception if no occurrence found
		if (columnIndex == -1) {
			throw new ColumnNotFoundException("The column " + column + " could not be found in the given .csv");
		}

		// Return result
		return columnIndex;
	}

	/**
	 * Reads the given column of a .csv file located at the given path and returns
	 * it as an array.
	 * 
	 * @param path
	 *            The path where the .csv file is located
	 * @param column
	 *            The column to read
	 * @return The values of the column of the file in an array
	 * @throws IOException
	 *             If an Exception occurs while trying to read the file
	 * @throws ColumnNotFoundException
	 *             If the column name does not occurr in the header of the fiven
	 *             file
	 */
	public static double[] getColumnValues(Path path, String column) throws IOException, ColumnNotFoundException {
		// Initialize values and reader
		List<Double> values = new ArrayList<Double>();
		BufferedReader reader = Files.newBufferedReader(path, Util.CHARSET);

		// Find index of measure
		String line = reader.readLine();
		int measureIndex = findIndexOfColumn(line, column);

		// Get contents
		while ((line = reader.readLine()) != null) {
			String[] contents = line.split(";");
			double value = Double.parseDouble(contents[measureIndex]);
			values.add(value);
		}

		return values.stream().mapToDouble(d -> d).toArray();
	}

	private static List<String> createListOfColumnTitles(Set<String> columnTitles) {
		// initialize list to return
		List<String> toReturn = new ArrayList<>();

		// ensure that data id is in the first column
		Set<String> columnTitlesCopy = new HashSet<String>(columnTitles);
		if (columnTitlesCopy.contains(Util.DATA_ID) && columnTitlesCopy.contains(Util.RANKER_BUILD_TIMES)) {
			// TODO there must be a better way for this -> don't assume always exist / data
			// id is first when deleting for analysis etc. ?
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

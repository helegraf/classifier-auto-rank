package experiments.two_part.part_two.output;

import java.io.FileWriter;
import java.io.IOException;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;

import jaicore.basic.SQLAdapter;

public class OutputHandler {

	public enum COLUMN {
		DATASET_ID, TRUE_RANKING, TRUE_VALUES, PREDICTED_RANKING, PREDICTED_VALUES, TRAINING_TIME, PREDICTION_TIME
	}

	private final String[] header;

	private List<String> datasetIds = new ArrayList<>();
	private List<List<String>> trueRankings = new ArrayList<>();
	private List<List<Double>> trueValues = new ArrayList<>();
	private List<List<String>> predictedRankings = new ArrayList<>();
	private List<List<Double>> predictedValues = new ArrayList<>();
	private List<Long> trainingTimes = new ArrayList<>();
	private List<Long> predictionTimes = new ArrayList<>();

	public OutputHandler(Map<COLUMN, String> columnNames) {
		header = new String[columnNames.size()];
		header[0] = columnNames.get(COLUMN.DATASET_ID);
		header[1] = columnNames.get(COLUMN.TRUE_RANKING);
		header[2] = columnNames.get(COLUMN.TRUE_VALUES);
		header[3] = columnNames.get(COLUMN.PREDICTED_RANKING);
		header[4] = columnNames.get(COLUMN.PREDICTED_VALUES);
		header[5] = columnNames.get(COLUMN.TRAINING_TIME);
		header[6] = columnNames.get(COLUMN.PREDICTION_TIME);
	}

	public OutputHandler(OutputConfig config) {
		header = new String[7];
		header[0] = config.getDatasetIdColumn();
		header[1] = config.getTrueRankingColumn();
		header[2] = config.getTrueValueColumn();
		header[3] = config.getPredictedRankingColumn();
		header[4] = config.getPredictedValuesColumn();
		header[5] = config.getTrainingTimeColumn();
		header[6] = config.getPredictTimeColumn();
	}

	public void addRecord(String datasetId, List<String> trueRanking, List<Double> trueValues,
			List<String> predictedRanking, List<Double> predictedValues, long trainingTime, long predictionTime) {
		this.datasetIds.add(datasetId);
		this.trueRankings.add(trueRanking);
		this.trueValues.add(trueValues);
		this.predictedRankings.add(predictedRanking);
		this.predictedValues.add(predictedValues);
		this.trainingTimes.add(trainingTime);
		this.predictionTimes.add(predictionTime);
	}

	public void writeFile(String location, String file) throws IOException {
		FileWriter out = new FileWriter(location + "/" + file);
		try (CSVPrinter printer = new CSVPrinter(out, CSVFormat.MYSQL.withHeader(header))) {
			for (int i = 0; i < datasetIds.size(); i++) {
				printer.printRecord(datasetIds.get(i), trueRankings.get(i), trueValues.get(i), predictedRankings.get(i),
						predictedValues.get(i), trainingTimes.get(i), predictionTimes.get(i));
			}
		}
	}

	public void uploadFile(SQLAdapter adapter, String location, String name, String table, int experimentId) throws SQLException {
		String sql = String.format(
				"LOAD DATA LOCAL INFILE \"%s\" into table %s IGNORE 1 LINES (%s,%s,%s,%s,%s,%s,%s) SET experiment_id=%d",
				location + "/" + name, table, header[0], header[1], header[2], header[3], header[4], header[5], header[6],
				experimentId);
		adapter.insertNoNewValues(sql, new ArrayList<Object>());
	}

	public String[] getHeader() {
		return header;
	}
}

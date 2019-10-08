package experiments.two_part;

import java.io.IOException;
import java.sql.SQLException;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;

import experiments.two_part.part_two.output.OutputHandler;
import experiments.two_part.part_two.output.OutputHandler.COLUMN;
import jaicore.basic.SQLAdapter;

public class OutputHandlerTest {

	private String host;
	private String user;
	private String pw;
	private String db;

	public void testCreateAndUploadFile() throws IOException, SQLException {
		Map<COLUMN, String> map = new EnumMap<>(COLUMN.class);
		map.put(COLUMN.DATASET_ID, "dataset_id");
		map.put(COLUMN.TRUE_RANKING, "true_ranking");
		map.put(COLUMN.TRUE_VALUES, "true_values");
		map.put(COLUMN.PREDICTED_RANKING, "predicted_ranking");
		map.put(COLUMN.PREDICTED_VALUES, "predicted_values");
		map.put(COLUMN.TRAINING_TIME, "training_time");
		map.put(COLUMN.PREDICTION_TIME, "prediction_time");

		OutputHandler handler = new OutputHandler(map);
		List<String> correctRanking = Arrays.asList("A", "B", "C");
		List<String> predictedRanking = Arrays.asList("C", "B", "A");
		List<Double> values = Arrays.asList(3.0, 2.0, 1.0);
		handler.addRecord("id1", correctRanking, values, predictedRanking, values, 1000, 1);
		handler.addRecord("id2", correctRanking, values, null, null, 1000, 0);

		handler.writeFile(".", "data.txt");

		try (SQLAdapter adapter = new SQLAdapter(host, user, pw, db)) {
			handler.createIntermediateResultsTable(adapter, db, "test");
			handler.uploadFile(adapter, ".", "data.txt", "test", 1);
		}
	}
}

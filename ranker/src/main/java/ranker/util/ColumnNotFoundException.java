package ranker.util;

/**
 * Indicated that a column name in the header of a .csv file was not found.
 * 
 * @author Helena Graf
 *
 */
public class ColumnNotFoundException extends Exception {

	/**
	 * Generated serial version UID
	 */
	private static final long serialVersionUID = 2468083773815872342L;

	/**
	 * Constructs an {@link ColumnNotFoundException} with {@code null} as its detail
	 * message.
	 */
	public ColumnNotFoundException() {
		super();
	}

	/**
	 * Constructs a new {@link ColumnNotFoundException} with the specified detail
	 * message.
	 * 
	 * @param message
	 */
	public ColumnNotFoundException(String message) {
		super(message);
	}

	/**
	 * Constructs a new {@link ColumnNotFoundException} with the specified cause and
	 * a detail message of {@code cause==null? null : cause.toString())} (which
	 * typically contains the class and detail message of cause)
	 * 
	 * @param cause
	 */
	public ColumnNotFoundException(Throwable cause) {
		super(cause);
	}

	/**
	 * Constructs a new {@link ColumnNotFoundException} with the specified detail
	 * message and cause.
	 * 
	 * @param message
	 * @param cause
	 */
	public ColumnNotFoundException(String message, Throwable cause) {
		super(message, cause);
		new Exception();
	}

}

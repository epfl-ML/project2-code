package ch.epfl.ml.csv

import java.io.Closeable
import java.io.OutputStream
import java.io.PrintStream

/** A writer for the CSV files. */
interface CSVWriter : Closeable {

  /** The names of the headers of the CSV file. */
  val columns: List<String>

  /** Writes the columns. */
  fun writeColumns(columns: List<String>)

  /** Writes a line to the CSV file. */
  fun write(values: List<String>)

  /** Removes the provided [columns] from the CSV file. */
  operator fun minus(columns: Set<String>): CSVWriter {
    return FilterCSVWriter(this.columns - columns, this)
  }
}

/**
 * A filter for a CSV writer, which can be used to filter out some values or to duplicate them.
 *
 * @param columns the columns of the filtered CSV.
 * @param writer the [CSVWriter] which is used to write the filtered values.
 */
class FilterCSVWriter(
    override val columns: List<String>,
    private val writer: CSVWriter,
) : CSVWriter {

  /** The indices of the resulting values. */
  private val indices = columns.map { writer.columns.indexOf(it) }.toIntArray()

  override fun writeColumns(columns: List<String>) {
    writer.writeColumns(columns)
  }

  override fun write(values: List<String>) {
    writer.write(indices.map { values[it] })
  }

  override fun close() = writer.close()
}

/** A [CSVWriter] which writes the epochs as CSV to an output file. */
class OutputStreamCSVWriter(
    override val columns: List<String>,
    stream: OutputStream,
) : CSVWriter {

  private val stream = PrintStream(stream)

  override fun writeColumns(columns: List<String>) {
    stream.println(columns.joinToString(","))
  }

  override fun write(values: List<String>) {
    for (i in values.indices) {
      stream.print(values[i])
      if (i != values.lastIndex) stream.print(',')
      if (i == values.lastIndex) stream.println()
    }
  }

  override fun close() = stream.close()
}

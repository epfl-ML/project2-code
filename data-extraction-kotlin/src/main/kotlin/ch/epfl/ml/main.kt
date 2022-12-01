@file:JvmName("Main")
@file:Suppress("LocalVariableName", "PropertyName")

package ch.epfl.ml

import ch.epfl.ml.args.Arguments
import ch.epfl.ml.csv.CSVWriter
import ch.epfl.ml.csv.OutputStreamCSVWriter
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.system.exitProcess

/**
 * A single value in the input stream.
 *
 * @param rawState the raw sleep phase.
 * @property state the sleep phase.
 * @param bin the DC components of the EEG.
 * @param EEGv the EEG variability.
 * @param EMGv the EMG variability.
 * @param temp the temperature of the mouse.
 */
class Epoch(
    private val rawState: Char,
    private val bin: FloatArray,
    private val EEGv: Float,
    private val EMGv: Float,
    private val temp: Float,
) {

  private val state: Char
    get() = mapPhase(rawState)

  fun features(): List<String> =
      listOf(
          rawState.toString(),
          state.toString(),
          EEGv.toString(),
          EMGv.toString(),
          temp.toString(),
          *bin.map { it.toString() }.toTypedArray(),
      )

  companion object Factory {

    val Features = listOf("rawState", "state", "EEGv", "EMGv", "temp", *Array(401) { "bin$it" })

    /** Reads the next epoch from the input stream. */
    fun from(bytes: InputStream): Epoch {
      val state = bytes.requireByte().toInt().toChar()
      val bin = FloatArray(401)
      for (i in bin.indices) {
        bin[i] = bytes.readFloat()
      }
      val EEGv = bytes.readFloat()
      val EMGv = bytes.readFloat()
      val temp = bytes.readFloat()
      return Epoch(rawState = state, bin = bin, EEGv = EEGv, EMGv = EMGv, temp = temp)
    }

    /** Maps the sleep [phase] to well-known values. */
    private fun mapPhase(phase: Char): Char {
      return when (phase) {
        'w',
        '1',
        '4' -> 'w'
        'n',
        '2',
        '5' -> 'n'
        'r',
        '3',
        '6' -> 'r'
        else -> throw IllegalArgumentException("Unknown phase $phase.")
      }
    }
  }
}

fun InputStream.requireByte(): Byte {
  val byte = read()
  if (byte == -1) {
    throw IllegalArgumentException("Unexpected end of stream.")
  }
  return byte.toByte()
}

/** Returns the next four bytes of this stream as a float. */
fun InputStream.readFloat(): Float {
  val bytes = ByteArray(4) { requireByte() }
  val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
  return buffer.getFloat(0)
}

/** Prints the instructions to use the program, and exits. */
fun printInstructions(): Nothing {
  println("Usage: java -jar ml.jar <input> <output>")
  println("Options: -i <input> -o <output> -col <c1,c2,c3> -bin <true|false> -sleep <true|false>")
  exitProcess(1)
}

/**
 * The ch.epfl.ml.main entry point of the application.
 *
 * @param args the command line arguments
 */
fun main(args: Array<String>) {
  val arguments = Arguments.parse(args) ?: printInstructions()
  val stream = FileInputStream(arguments.inputFilePath).buffered()
  var writer: CSVWriter =
      OutputStreamCSVWriter(
          columns = Epoch.Features,
          stream = FileOutputStream(arguments.outputFilePath).buffered(),
      )
  val removed = (Epoch.Features.toSet() - arguments.columns.toSet()).toMutableSet()
  if (!arguments.includeBin) removed += Array(401) { "bin$it" }
  if (!arguments.includeRawSleepPhase) removed += "rawState"
  writer -= removed

  println("Keeping ${Epoch.Features - removed}.")
  println("Removing $removed.")
  var count = 0
  writer.writeColumns(writer.columns)
  while (true) {
    if (++count % 1000 == 0) println("Processed $count epochs.")
    try {
      writer.write(Epoch.from(stream).features())
    } catch (e: IllegalArgumentException) {
      break
    }
  }
  writer.close()
}

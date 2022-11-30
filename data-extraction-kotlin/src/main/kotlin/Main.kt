@file:JvmName("Main")
@file:Suppress("LocalVariableName", "PropertyName")

import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.InputStream
import java.nio.ByteBuffer

/**
 * A single value in the input stream.
 *
 * @param state the sleep phase.
 * @param bin the DC components of the EEG.
 * @param EEGv the EEG variability.
 * @param EMGv the EMG variability.
 * @param temp the temperature of the mouse.
 */
class Epoch(
    val state: Char,
    val bin: FloatArray,
    val EEGv: Float,
    val EMGv: Float,
    val temp: Float,
) {

  companion object Factory {

    /** Reads the next epoch from the input stream. */
    fun from(bytes: InputStream): Epoch {
      val state = bytes.requireByte().toChar()
      val bin = FloatArray(401)
      for (i in bin.indices) {
        bin[i] = bytes.readFloat()
      }
      val EEGv = bytes.readFloat()
      val EMGv = bytes.readFloat()
      val temp = bytes.readFloat()
      return Epoch(mapPhase(state), bin, EEGv, EMGv, temp)
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
  val buffer = ByteBuffer.wrap(bytes)
  return buffer.getFloat(0)
}

/**
 * The main entry point of the application.
 *
 * @param args the command line arguments
 */
fun main(args: Array<String>) {
  if (args.size != 2) {
    println("Usage: <input file> <output file>")
    return
  }

  val stream = FileInputStream(args[0]).buffered()
  val out = FileOutputStream(args[1]).bufferedWriter()
  out.write("state,bin,EEGv,EMGv,temp\n")
  var count = 0
  while (true) {
    count++
    if (count % 1000 == 0) {
      println("Processed $count epochs.")
    }
    try {
      val epoch = Epoch.from(stream)
      out.write("${epoch.state},${epoch.EEGv},${epoch.EMGv},${epoch.temp}\n")
    } catch (e: IllegalArgumentException) {
      break
    }
  }
}

package ch.epfl.ml.args

import ch.epfl.ml.Epoch

/** The arguments of the command line, which will be applied by program when it runs. */
data class Arguments(
    val inputFilePath: String,
    val outputFilePath: String,
    val columns: List<String>,
    val includeBin: Boolean,
    val includeRawSleepPhase: Boolean,
) {

  companion object Factory {

    /**
     * Parses the command-line arguments.
     *
     * @param args the application arguments.
     */
    fun parse(args: Array<String>): Arguments? {
      if (args.size < 2) return null

      var input = args[0]
      var output = args[1]
      var columns = Epoch.Features
      var includeBin = false
      var includeRawSleepPhase = false

      // Parse all pairs of flags, and keep the last one for each flag.
      var index = 2
      while (index < args.size) {
        val flag = args[index++]
        if (index == args.size) return null
        val value = args[index++]
        when (flag) {
          "-i" -> input = value
          "-o" -> output = value
          "-col" -> columns = value.split(",")
          "-bin" -> includeBin = value.toBooleanStrictOrNull() ?: return null
          "-sleep" -> includeRawSleepPhase = value.toBooleanStrictOrNull() ?: return null
          else -> return null
        }
      }

      return Arguments(
          inputFilePath = input,
          outputFilePath = output,
          columns = columns,
          includeBin = includeBin,
          includeRawSleepPhase = includeRawSleepPhase,
      )
    }
  }
}

# Instructions

Using [Gradle](https://gradle.org) (requires Java 17), you can run the following command to extract the data from the
database and generate the CSV files:

```shell
./gradlew run --args="../data/SMO_files/10101.smo out2.csv"
```

The following arguments are available:

- `-i <path>` to override the input file (still requires to specify the
  input file as the first argument, which will then be ignored)
- `-o <path>` to override the output file (still requires to specify the
  output file as the second argument, which will then be ignored)
- `-bin <true|default=false>` to override whether bin features should be
  extracted
- `-sleep <true|default=false>` to override whether raw sleep features
  should be extracted
- `-col <a,b,c>` to override the columns to extract (defaults to all)

The shell script `extract_all.sh` can be used to apply the extraction script to the
contents of a whole folder.

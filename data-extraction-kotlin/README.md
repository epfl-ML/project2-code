# Data extraction

Using Gradle, you can run the following command to extract the data from the
database and generate the CSV files:

```shell
./gradlew run --args="../data/SMO_files/10101.smo out2.csv"
```

The following arguments are possible:

- `-i <path>` to override the input file
- `-o <path>` to override the output file
- `-bin <true|default=false>` to override whether bin features should be
  extracted
- `-sleep <true|default=false>` to override whether raw sleep features
  should be extracted
- `-col <a,b,c>` to override the columns to extract (defaults to all)

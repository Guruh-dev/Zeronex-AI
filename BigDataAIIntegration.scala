import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import scalaj.http._

object MyAIIntegration {
  def main(args: Array[String]): Unit = {
    // Replace with the URL of your AI endpoint
    val aiEndpoint = "http://localhost:5000/predict"
    
    // Example input data for your AI; update this JSON as needed
    val inputData = """{"input": "sample data"}"""
    
    try {
      // Send a POST request to your AI endpoint with the JSON payload
      val response = Http(aiEndpoint)
        .postData(inputData)
        .header("Content-Type", "application/json")
        .asString

      // Print out the response received from the AI service
      println("Response from AI:")
      println(response.body)
    } catch {
      case e: Exception =>
        println("Error integrating with AI: " + e.getMessage)
    }
  }
}

object BigDataAIIntegration {
  def main(args: Array[String]): Unit = {
    // Create a Spark session
    val spark = SparkSession.builder()
      .appName("Big Data AI Integration")
      // For local testing, you can uncomment the following line:
      // .master("local[*]")
      .getOrCreate()

    // Import implicits for easy DataFrame operations
    import spark.implicits._

    // Load data from CSV file.
    // The CSV file is expected to have a header and columns such as "label", "feature1", "feature2", "feature3".
    // Adjust the file path and column names as needed.
    val data = spark.read
      .option("inferSchema", "true")
      .option("header", "true")
      .csv("path/to/your/data.csv")

    // Assemble features into a single vector column named "features"
    val assembler = new VectorAssembler()
      .setInputCols(Array("feature1", "feature2", "feature3"))
      .setOutputCol("features")

    // Convert string labels to numeric values using StringIndexer.
    // This is useful when your label column is categorical.
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(data)

    // Split data into training (80%) and testing (20%) sets
    val Array(trainingData, testData) = data.randomSplit(Array(0.8, 0.2), seed = 1234L)

    // Define the Logistic Regression model.
    // Here, we use logistic regression for binary classification.
    val lr = new LogisticRegression()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("features")
      .setMaxIter(10)

    // Create a Pipeline that chains the data processing and modeling stages.
    val pipeline = new Pipeline().setStages(Array(assembler, labelIndexer, lr))

    // Build a parameter grid for hyperparameter tuning.
    // This grid tests different regularization parameters and elastic net mixing.
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.01, 0.1, 0.5))
      .addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0))
      .build()

    // Define an evaluator using the area under ROC metric.
    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setMetricName("areaUnderROC")

    // Set up cross-validation to tune hyperparameters.
    // This example uses 3-fold cross-validation.
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)

    // Train the model using cross-validation on the training dataset.
    val cvModel = cv.fit(trainingData)

    // Make predictions on the test dataset.
    val predictions = cvModel.transform(testData)

    // Evaluate the model performance using the evaluator.
    val auc = evaluator.evaluate(predictions)
    println(s"Area Under ROC on test data = $auc")

    // Display some example predictions with their features, actual label, predicted label, and probabilities.
    predictions.select("features", "indexedLabel", "prediction", "probability").show(10)

    // Stop the Spark session.
    spark.stop()
  }
}

#!/usr/bin/env Rscript
# MyAdvancedAIIntegration.R
#
# This advanced script integrates with an AI service for complex features.
# It reads input data from a CSV file, uses command-line options for configuration,
# implements exponential backoff retry for the HTTP POST request, and logs responses.
#
# Required packages:
#   - httr: For HTTP requests
#   - jsonlite: For JSON encoding/decoding
#   - optparse: For command-line argument parsing
#
# Install required packages if needed:
#   install.packages("httr")
#   install.packages("jsonlite")
#   install.packages("optparse")
#
# Advance AI Integration Script Zeronex AI Credits by: Guruh-dev
library(httr)
library(jsonlite)
library(optparse)

# Define command line options
option_list <- list(
  make_option(c("-f", "--file"), type="character", default=NULL, 
              help="Path to CSV file containing input data", metavar="character"),
  make_option(c("-e", "--endpoint"), type="character", default=Sys.getenv("AI_ENDPOINT"),
              help="AI service endpoint URL (default from environment variable AI_ENDPOINT)", metavar="character")
)
opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

# Validate inputs
if (is.null(opt$file)) {
  stop("Input CSV file must be provided using -f or --file option.")
}
if (is.null(opt$endpoint) || opt$endpoint == "") {
  stop("AI service endpoint URL must be provided either as an option or via the AI_ENDPOINT environment variable.")
}

# Read input data from CSV file
input_data <- read.csv(opt$file, stringsAsFactors = FALSE)
# Assumes the CSV contains at least one column (e.g., "input") for the AI predictions

# Function to call the AI service with an exponential backoff retry mechanism
call_ai_service <- function(url, payload, max_retries = 5) {
  attempt <- 1
  repeat {
    tryCatch({
      response <- POST(url, body = payload, add_headers("Content-Type" = "application/json"))
      if (status_code(response) == 200) {
        return(content(response, as = "parsed", encoding = "UTF-8"))
      } else {
        warning(sprintf("Attempt %d: Received HTTP %d", attempt, status_code(response)))
        stop("Non-200 status received")
      }
    }, error = function(e) {
      if (attempt >= max_retries) {
        stop(sprintf("Failed after %d attempts: %s", attempt, e$message))
      } else {
        wait_time <- 2^(attempt)  # exponential backoff: 2, 4, 8... seconds
        message(sprintf("Attempt %d failed: %s. Retrying in %d seconds...", attempt, e$message, wait_time))
        Sys.sleep(wait_time)
        attempt <<- attempt + 1
      }
    })
  }
}

# Process each row from the CSV and call the AI service
results <- list()
for (i in seq_len(nrow(input_data))) {
  # Convert the current row to a list (adjust the structure as needed)
  row_input <- as.list(input_data[i,])
  # Convert input data to JSON format
  json_payload <- toJSON(row_input, auto_unbox = TRUE)
  
  # Log the payload being sent
  message(sprintf("Processing row %d: %s", i, json_payload))
  
  # Attempt to call the AI service with the JSON payload
  result <- tryCatch({
    call_ai_service(opt$endpoint, json_payload)
  }, error = function(e) {
    list(error = e$message)
  })
  
  # Save the result for this row
  results[[i]] <- result
}

# Save all AI responses to a JSON file for further processing
output_file <- "ai_results.json"
write(toJSON(results, pretty = TRUE, auto_unbox = TRUE), file = output_file)
message(sprintf("AI results have been saved to %s", output_file))

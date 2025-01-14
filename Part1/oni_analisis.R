library(forecast)
library(ggplot2)
library(zoo)
library (gbm)
library(Metrics)
library(dplyr)
library(caret)
library(tibble)
library(data.table)
library(splines)


######## OCEANIC NIÑO INDEX ##########
# Load the data
oni_df <- read.csv("ONI_data.csv")

# View the first rows
str(oni_df)

# Discard the row with null values
oni_df <- na.omit(oni_df)

# Prepare the data
oni_df <- data.frame(
  date = as.Date(oni_df$time),
  year = oni_df$year,
  month = oni_df$month,
  oni = oni_df$sst 
)

# Convert date to numeric (years and months)
oni_df$year <- as.numeric(format(oni_df$date, "%Y"))
oni_df$month <- as.numeric(format(oni_df$date, "%m"))

# View the data again
str(oni_df)

# Extract the 'oni' column
oni_values <- oni_df$oni

# Define the start date (adjust year and month as needed)
start_year <- as.numeric(format(min(oni_df$date), "%Y"))
start_month <- as.numeric(format(min(oni_df$date), "%m"))

# Convert to a time series object
oni_ts <- ts(oni_values, start = c(start_year, start_month), frequency = 12)

# Split data: 80% training and 20% testing
train_size <- floor(0.8 * length(oni_ts))
train <- window(oni_ts, end = c(time(oni_ts)[train_size]))
test <- window(oni_ts, start = c(time(oni_ts)[train_size + 1]))

# MODEL 1: ARIMA MODELS
###### MODEL 1: ARIMA(3,0,2) without seasonlity
arima1 <- Arima(oni_ts, order = c(3,0,2))
AIC(arima1)
BIC(arima1)
summary(arima1)

###### MODEL 2: SARIMA(3,0,2) with year seasonality
arima2 <- Arima(oni_ts, order = c(3,0,2), seasonal = c(1,1,1))
AIC(arima2)
BIC(arima2)

###### MODEL 3: SARIMA(3,0,2) with 5-year cycle seasonality
oni_ts2 <- ts(oni_values, frequency = 5)

arima3 <- Arima(oni_ts2, order = c(3,0,2), seasonal = c(1,1,1))
AIC(arima3)
BIC(arima3)

###### MODEL 4: LR+SARIMA(3,0,2) WITH 5-year cycle seasonality
mod<- tslm(oni_ts~ trend+season) 

mod_residuals <- residuals(mod)
residuals_ts <- ts(mod_residuals, frequency = 5)

arima4 <- Arima(residuals_ts, order = c(3, 0, 2), seasonal = c(1, 1, 1))
AIC(arima4)
BIC(arima4)

###### MODEL 5: LR+ARIMA(3,0,2)
arima5 <- Arima(residuals(mod), order = c(3, 0, 2))
AIC(arima5)
BIC(arima5)

###### MODEL 6: AUTO.ARIMA
arima6 <- auto.arima(oni_ts)
AIC(arima6)
BIC(arima6)

# MODEL 2: GBOOST 
# Lagged features
oni_df$oni_lag1 <- shift(oni_df$oni, n = 1, type = "lag")

# Train-test split
set.seed(1)  # For reproducibility
train_index <- createDataPartition(oni_df$oni, p = 0.8, list = FALSE)
train_data <- oni_df[train_index, ]
test_data <- oni_df[-train_index, ]

# Train GBM model
gbm_model <- gbm(
  formula = oni ~ year + month + oni_lag1,
  distribution = "gaussian",
  data = train_data,
  n.trees = 1000,
  shrinkage = 0.01,
  interaction.depth = 6,
  n.minobsinnode = 10,
  bag.fraction = 0.8,
  train.fraction = 0.8,
  cv.folds = 5,
  verbose = TRUE
)

# Assuming test_data is aligned with the test portion of the time series
# Extract the start and frequency from the test portion of the time series
start_time <- start(test)
freq <- frequency(test)

# Convert predictions_gboost to a time series
predictions_gboost <- ts(
  predict(gbm_model, newdata = test_data, n.trees = gbm.perf(gbm_model, method = "cv")),
  start = start_time,
  frequency = freq
)


## Let us now forcast using GBM

forecast_horizon <- 2 * 12# Number of months to forecast

forecast_data <- tail(test_data, 1)  # Start with the last row of test data
future_predictions <- numeric(forecast_horizon)  # Store predictions

for (i in 1:forecast_horizon) {
  # Predict using the most recent data (ensure it’s a single row)
  pred <- predict(gbm_model, newdata = forecast_data[nrow(forecast_data), , drop = FALSE], 
                  n.trees = gbm.perf(gbm_model, method = "cv"))
  
  # Save the prediction
  future_predictions[i] <- pred
  
  # Update forecast_data with the predicted value
  new_row <- forecast_data[nrow(forecast_data), , drop = FALSE]  # Clone the last row structure
  new_row$oni <- pred  # Replace `oni` with the predicted value
  new_row$oni_lag1 <- forecast_data$oni[nrow(forecast_data)]  # Use the last observed/predicted value as lag
  
  # Shift the month and year forward
  new_row$month <- new_row$month + 1
  if (new_row$month > 12) {
    new_row$month <- 1
    new_row$year <- new_row$year + 1
  }
  
  # Append the new_row to forecast_data for the next iteration
  forecast_data <- rbind(forecast_data, new_row)
}

forecast_ts <- ts(
  future_predictions,
  start = c(test_data$year[nrow(test_data)], test_data$month[nrow(test_data)]),
  frequency = 12
)


plot(
  ts(oni_df$oni, start = c(min(oni_df$year), min(oni_df$month)), frequency = 12),
  col = "blue",
  main = "2-Year ONI Forecast",
  lwd = 1.5,
  xlab = "Year",
  ylab = "ONI",
  xlim = c(min(oni_df$year), max(oni_df$year) + 2)  # Extend x-axis for forecast
)
lines(forecast_ts, col = "red", lty = 4)  # Add forecast as a dashed red line
legend("topleft", legend = c("Actual", "Forecast"), col = c("blue", "red"), lty = c(1, 2))


##

# Compute residuals
residuals <- test_data$oni - predictions

# Number of observations (n)
n <- nrow(test_data)

# Number of parameters (k)
# For GBM, we approximate k based on the number of trees and interaction depth
k <- gbm_model$interaction.depth * gbm_model$n.trees

# Residual Sum of Squares (RSS)
rss <- sum(residuals^2)

# Residual Variance (sigma^2)
sigma_squared <- rss / (n - k)

# Compute metrics
aic <- n * log(rss / n) + 2 * k
aicc <- aic + (2 * k * (k + 1)) / (n - k - 1)
bic <- n * log(rss / n) + k * log(n)
me <- mean(residuals)
rmse <- sqrt(mean(residuals^2))
mae <- mean(abs(residuals))

# Print metrics
cat("AIC:", aic, "\n")
cat("AICc:", aicc, "\n")
cat("BIC:", bic, "\n")
cat("Residual Variance (σ²):", sigma_squared, "\n")
cat("Mean Error (ME):", me, "\n")
cat("Root Mean Square Error (RMSE):", rmse, "\n")
cat("Mean Absolute Error (MAE):", mae, "\n")

# Predict on the entire dataset
full_predictions <- predict(
  gbm_model, 
  newdata = oni_df, 
  n.trees = gbm.perf(gbm_model, method = "cv")
)

# Add predictions to the dataset
oni_df$gboost <- full_predictions

# Use pretty() to generate better breaks for the y-axis
y_ticks <- pretty(c(min(y_range) - 0.2, max(y_range) + 3))

# Enhanced Plot: Actual vs Predicted ONI
plot(oni_df$date, oni_df$oni, type = "l", col = "blue", lwd = 2,
     xlab = "Date", ylab = "ONI",
     ylim = c(min(y_ticks), max(y_ticks)))  # Adjust y-axis limits based on pretty()

# Add the predicted ONI values
lines(oni_df$date, oni_df$gboost, col = "red", lwd = 2)

# Overlay points on predicted values for clarity
points(oni_df$date, oni_df$predicted_oni, col = "red", pch = 16)

# Add a legend
legend("topright", legend = c("Actual ONI", "Predicted ONI"),
       col = c("blue", "red"), lwd = c(2, 2), lty = c(1, 1))



#### MODEL 3: LINEAR REGRESSION+ARIMA
# Fit a regression model with trend, season, and ONI as external variable
mod<- tslm(oni_ts~ trend+season) 

# View the summary
summary(mod)

# Plot the ONI data
plot(oni_ts, col = "black", lwd = 1.5, ylab = "ONI Index", main = "Linear Regression Model Fit", xlab = "Time")

# Plot fitted model
lines(fitted(mod), col = "blue", lwd = 2, lty = 2)

# Add a legend
legend("topleft", legend = c("Precipitation Data", "Fitted Values"), 
       col = c("black", "blue"), lty = c(1, 2, 2), lwd = 2)

# Plot the residuals
tsdisplay(residuals(mod))

# Autocorrelation is pretty obvious

# Fit an ARIMA model in the residuals
arima_res<- Arima(residuals(mod), order = c(2,2,2))
arima_res<--auto.arima(residuals(mod))

# Plot everything again
plot(oni_ts, col = "black", lwd = 1.5, ylab = "Precipitations", main = "Linear Regression+ARIMA Model Fit", xlab = "Time")
lines(fitted(mod)+fitted(arima_res), col = "blue", lwd = 2, lty = 2)
legend("topleft", legend = c("Precipitation Data", "Fitted Values"), 
       col = c("black", "blue"), lty = c(1, 2, 2), lwd = 2)

# View the summary of the ARIMA on the residuals
summary(mod)
summary(arima_res)

# Combine the fitted values
combined_fitted <- fitted(mod) + fitted(arima_res)

# Add the values of the model in dataset
oni_df$lrarima <- as.numeric(combined_fitted)

# Check the residuals
checkresiduals(arima_res)

# NB: I tried to do this same procedure using Anual cycle + El Niño/La Niña
# cycle but turns out that just using Anual cycle gives better results,
# so I will not consider the other one.

# ANALYSIS: Looks like LINEAR REGRESSION+ARIMA (RMSE=0.0939) is better 
# than  ARIMA (RMSE=0.1077) and GBM (RMSE=0.1597).

#### MODEL 4: SPLINES
# Fit the model
fit <- smooth.spline(as.numeric(oni_df$date), oni_ts)

# Plotting
plot(oni_df$date, oni_df$oni, col = "black", lwd = 1.5, ylab = "ONI Index", main = "Spline Model Fit", xlab = "Time")
lines(fit, col = "blue", lwd = 2, lty = 2)
legend("topleft", legend = c("Original Data", "Spline Fit"), col = c("black", "blue"), lty = 1, lwd = 2)

# View the summary 
summary(fit)

# Extract residuals
residuals <- fit$yin - fit$y
n <- length(fit$yin)  # Sample size
k <- fit$df          # Effective degrees of freedom

#
cat("Number of splines (df):", k, "\n")

# Compute Residual Sum of Squares (RSS)
rss <- sum(residuals^2)

# Compute metrics
sigma_squared <- rss / (n - k) # Residual variance (sigma^2)
aic <- n * log(rss / n) + 2 * k
bic <- n * log(rss / n) + log(n) * k
aicc <- aic + (2 * k * (k + 1)) / (n - k - 1)  # Corrected AIC
me <- mean(residuals)  # Mean Error
rmse <- sqrt(mean(residuals^2))  # Root Mean Square Error
mae <- mean(abs(residuals))  # Mean Absolute Error

# Print metrics
cat("Residual Variance:", sigma_squared, "\n")
cat("AIC:", aic, "\n")
cat("AICc:", aicc, "\n")
cat("BIC:", bic, "\n")
cat("Mean Error (ME):", me, "\n")
cat("Root Mean Square Error (RMSE):", rmse, "\n")
cat("Mean Absolute Error (MAE):", mae, "\n")

# Add it to the dataset
fitted_values <- predict(fit)
oni_df$splines <- as.numeric(fitted_values$y)

# Save in a file with all the data
write.csv(oni_df, "oni_df.csv", row.names = TRUE)

# Final plot of everything

# Plot the results
autoplot(oni_ts) +
  autolayer(forecast_values$mean, series = "ARIMA(3,0,2)", PI = FALSE) +
  autolayer(test, series = "Test Data") +
  xlab("Time") +
  ylab("ONI Index") +
  theme_minimal()

######## SOUTHERN OSCILATION INDEX ##########
# Load the dataset
soi_df <- read.csv("SOI_data.csv")

# Discard the row with null values
soi_df <- na.omit(soi_df)

# Prepare the data
soi_df <- data.frame(
  date = as.Date(soi_df$time),
  soi = soi_df$SOI,
  oni = soi_df$sst
)

# Define start year and month
start_year <- as.numeric(format(min(soi_df$date), "%Y"))
start_month <- as.numeric(format(min(soi_df$date), "%m"))

# Convert to time series
soi_ts <- ts(soi_df$soi, start = c(start_year, start_month), frequency = 12)
oni_ts <- ts(soi_df$oni, start = c(start_year, start_month), frequency = 12)

#### MODEL 1: ARIMA
# Fit ARIMA model
arima_model <- Arima(soi_ts, order = c(0,1,2))
arima_model <- Arima(soi_ts, order = c(3,0,2), seasonal = c(2,0,0))

summary(arima_model)

# Plot the SOI data
plot(soi_ts, col = "black", lwd = 1.5, ylab = "SOI Index", main = "ARIMA Model Fit", xlab = "Time")

# Fitted values
fitted_values <- fitted(arima_model)

# Add fitted values
lines(fitted_values, col = "blue", lwd = 2, lty = 2)
# Add a legend
legend("topleft", legend = c("SOI Data", "Fitted Values"), 
       col = c("black", "blue"), lty = c(1, 2, 2), lwd = 2)

# View the summary of the model
summary(arima_model)

# Add this to the dataset
soi_df$arima <- as.numeric(fitted_values)

#### MODEL 2: ARIMAX
# Check the correlation between ONI and SOI
correlation <- cor(oni_ts, soi_ts, use = "complete.obs")
print(paste("Pearson correlation:", correlation))

# Fit ARIMAX model using ONI as external variable
arimax_model <- arima(soi_ts, order = c(0,1,2), xreg= oni_ts)
arimax_model <- auto.arima(soi_ts, xreg= oni_ts)

summary(arimax_model)

# Plot the SOI data
plot(soi_ts, col = "black", lwd = 1.5, ylab = "SOI Index", main = "ARIMAX Model Fit", xlab = "Time")

# Fitted values
fitted_values <- fitted(arimax_model)

# Add fitted values
lines(fitted_values, col = "blue", lwd = 2, lty = 2)
# Add a legend
legend("topleft", legend = c("SOI Data", "Fitted Values"), 
       col = c("black", "blue"), lty = c(1, 2, 2), lwd = 2)

# View model summary
summary(arimax_model)

# Check the residuals
checkresiduals(arimax_model)

# Add it to the dataset
soi_df$arimax <- as.numeric(fitted_values)

# ANALYSIS: All the metrics used (AIC, BIC, RMSE, MAE, and sigma^2) 
# show better performance using ARIMAX, which makes a lot of sense
# since the correlation between ONI and SOI is very high (-0.6977). 
# Take for instance the metric AIC for ARIMA (AIC=883.86) and 
# ARIMAX (AIC=806.59), ARIMAX shows a significantly lower AIC, which 
# suggests it fits the data better.

#### MODEL 3: LINEAR REGRESSION + ARIMAX
# Fit a regression model with trend, season, and ONI as external variable
mod<- tslm(soi_ts~ trend+season) 

# View the summary
summary(mod)

# Plot the ONI data
plot(soi_ts, col = "black", lwd = 1.5, ylab = "SOI Index", main = "Linear Regression Model Fit", xlab = "Time")

# Plot fitted model
lines(fitted(mod), col = "blue", lwd = 2, lty = 2)

# Add a legend
legend("topleft", legend = c("Precipitation Data", "Fitted Values"), 
       col = c("black", "blue"), lty = c(1, 2, 2), lwd = 2)

# Plot the residuals
tsdisplay(residuals(mod))

# Autocorrelation is pretty obvious

# Fit an ARIMA model in the residuals
arimax_res<- auto.arima(residuals(mod), xreg=oni_ts)
arimax_res<- Arima(residuals(mod), order = c(3,0,2), xreg=oni_ts)
summary(arimax_res)

# Plot everything again
plot(soi_ts, col = "black", lwd = 1.5, ylab = "SOI Index", main = "Linear Regression+ARIMAX Model Fit", xlab = "Time")
lines(fitted(mod)+fitted(arimax_res), col = "blue", lwd = 2, lty = 2)
legend("topleft", legend = c("SOI Data", "Fitted Values"), 
       col = c("black", "blue"), lty = c(1, 2, 2), lwd = 2)

# View the summary of the ARIMA on the residuals
summary(mod)
summary(arimax_res)

# Check the residuals
checkresiduals(arimax_res)

# Add it to the dataset
soi_df$lrarima <- as.numeric(fitted(mod)+fitted(arimax_model))

# Save the dataset in a file
write.csv(soi_df, "soi_df.csv", row.names = TRUE)

######## PRECIPITATIONS ##########
# Load the dataset
prep_df <- read.csv("precipitation_data.csv")

# View the first rows
str(prep_df)

# Prepare the data
prep_df <- data.frame(
  date = as.Date(prep_df$time),
  prep = prep_df$precipitation,
  oni = prep_df$sst
)

# Discard the row with null values
prep_df <- na.omit(prep_df)

# Define start year and month
start_year <- as.numeric(format(min(prep_df$date), "%Y"))
start_month <- as.numeric(format(min(prep_df$date), "%m"))

# Convert to time series
prep_ts <- ts(prep_df$prep, start = c(start_year, start_month), frequency = 12)
oni_ts <- ts(prep_df$oni, start = c(start_year, start_month), frequency = 12)

#### MODEL 1: ARIMA 
# Fit ARIMA model
arima_model <- auto.arima(prep_ts, seasonal = TRUE)

# Plot the Precipitation data
plot(prep_ts, col = "black", lwd = 1.5, ylab = "Precipitations", main = "ARIMA Model Fit", xlab = "Time")

# Fitted values
fitted_values <- fitted(arima_model)

# Add fitted values
lines(fitted_values, col = "blue", lwd = 2, lty = 2)
# Add a legend
legend("topleft", legend = c("Precipitation Data", "Fitted Values"), 
       col = c("black", "blue"), lty = c(1, 2, 2), lwd = 2)

# View the summary of the model
summary(arima_model)

# Add it to the dataset 
prep_df$arima <- as.numeric(fitted_values)

# Check the correlation between ONI and Precipitations
correlation <- cor(oni_ts, prep_ts, use = "complete.obs")
print(paste("Pearson correlation:", correlation))

#### MODEL 2: ARIMAX
# Fit ARIMAX model using ONI as external variable
arimax_model <- auto.arima(prep_ts, xreg = oni_ts)

# Plot the Precipitation data
plot(prep_ts, col = "black", lwd = 1.5, ylab = "Precipitations in Mallares (Peru)", main = "ARIMAX Model Fit", xlab = "Time")

# Fitted values
fitted_values <- fitted(arimax_model)

# Add fitted values
lines(fitted_values, col = "blue", lwd = 2, lty = 2)
# Add a legend
legend("topleft", legend = c("SOI Data", "Fitted Values"), 
       col = c("black", "blue"), lty = c(1, 2, 2), lwd = 2)
# View model summary
summary(arimax_model)

# Check the residuals
checkresiduals(arimax_model)

# Add it to the dataset 
prep_df$arimax <- as.numeric(fitted_values)

# ANALYSIS: Also in this case, all the metrics show a better performance
# of the ARIMAX model. This time though, the improvement between 
# ARIMA (AIC=1540.31) and ARIMAX (AIC=1534.2) is not as evident as 
# the previous case. The reason behind this is the correlation 
# between ONI and Precipitation, which is much lower this time (0.2673).

#### MODEL 3: LINEAR REGRESSION+ARIMA 
# Fit a regression model with trend, season, and ONI as external variable
mod<- tslm(prep_ts~ trend+season) 

# View the summary
summary(mod)

# Prol the Precipitation data
plot(prep_ts, col = "black", lwd = 1.5, ylab = "Precipitations", main = "Linear Regression Model Fit", xlab = "Time")

# Plot fitted model
lines(fitted(mod), col = "blue", lwd = 2, lty = 2)

# Add a legend
legend("topleft", legend = c("Precipitation Data", "Fitted Values"), 
       col = c("black", "blue"), lty = c(1, 2, 2), lwd = 2)

# Plot the residuals
tsdisplay(residuals(mod))

# Autocorrelation is pretty obvious

# Fit an ARIMA model in the residuals
arimax_res<- auto.arima(residuals(mod), xreg = oni_ts)

# Plot everything again
plot(prep_ts, col = "black", lwd = 1.5, ylab = "Precipitations", main = "Linear Regression+ARIMA Model Fit", xlab = "Time")
lines(fitted(mod)+fitted(arimax_res), col = "blue", lwd = 2, lty = 2)
legend("topleft", legend = c("Precipitation Data", "Fitted Values"), 
       col = c("black", "blue"), lty = c(1, 2, 2), lwd = 2)

# View the summary of the ARIMA on the residuals
summary(mod)
summary(arimax_res)

# Add it to the dataset
prep_df$lrarima <- as.numeric(fitted(mod)+fitted(arimax_res))

# ANALYSIS 2: Looks like LINEAR REGRESSION+ARIMA is better than the 
# previous models, AIC=1498.72.
# Add it to the dataset 

# Save the dataset in a file
write.csv(prep_df, "prep_df.csv", row.names = TRUE)

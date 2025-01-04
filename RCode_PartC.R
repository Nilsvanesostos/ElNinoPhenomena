install.packages("lubridate")  # Run this if you haven't installed it yet
library(lubridate)

# Load necessary libraries
library(forecast)
library(ggplot2)
library(tseries)

# Load the data
data <- read.csv("C:/Users/kocak/Desktop/Time-series/spbvl-oni-manual-v1.csv")
#data <- read.csv("C:/Users/kocak/Desktop/Time-series/spbvl-oni-manual-v2.csv")
#data <- read.csv("C:/Users/kocak/Desktop/Time-series/spbvl-oni-global.csv")

spbvl_std_ts <- ts(data$close_std, start = c(year(min(data$date)), month(min(data$date))), frequency = 12)
oni_std_ts <-  ts(data$oni_std, start = c(year(min(data$date)), month(min(data$date))), frequency = 12)
oni_std_sst_ts <- ts(data$sst_std, start = c(year(min(data$date)), month(min(data$date))), frequency = 12)
# Plot the first time series
plot(oni_std_ts, type = "l", col = "blue", lwd = 2, 
     ylab = "Values", xlab = "Time", 
     main = "SPBVL and ONI Time Series")
# Add the second time series to the same plot
lines(spbvl_std_ts, col = "red", lwd = 2)
# Add a legend
legend("topright", legend = c("ONI", "SPBVL"), 
       col = c("blue", "red"), lwd = 2)



# Align time series (ensure they have the same length)
#min_length <- min(length(oni_std_sst_ts), length(spbvl_std_ts))
min_length <- min(length(oni_std_ts), length(spbvl_std_ts))
#oni_aligned <- window(oni_std_sst_ts, end = time(oni_std_sst_ts)[min_length])
oni_aligned <- window(oni_std_ts, end = time(oni_std_ts)[min_length])
spbvl_aligned <- window(spbvl_std_ts, end = time(spbvl_std_ts)[min_length])

###################### Analysis without external variable (ONI) ##############

#------------------------------ BM ---------------------------------------
library(DIMORA)
#data <- read.csv("C:/Users/kocak/Desktop/Time-series/spbvl-oni-manual-v1.csv")

data$spbvl_cumulative <- cumsum(data$close)  # Convert to cumulative if applicable
plot(data$spbvl_cumulative)

bm_bvl<-BM(data$spbvl_cumulative,display = T)

summary(bm_bvl)
plot(bm_bvl)
future_time <- (1:nrow(data)) + (1:(365 * 2))

# Predict future values
predictions <- predict(bm_bvl, newx = future_time)

# Combine predictions with historical data for plotting
plot(1:nrow(data), data$spbvl_cumulative, type = "l", col = "blue",
     main = "Stock Prediction for the Next 2 Years", xlab = "Time", ylab = "Cumulative Close")
lines(future_time, predictions, col = "red", lty = 2)

legend("topleft", legend = c("Historical", "Predicted"), col = c("blue", "red"), lty = 1:2)

#------------------------- GGM --------------------------------------------
# Load the data
#data <- read.csv("C:/Users/kocak/Desktop/Time-series/spbvl-oni-manual-v1.csv")

# Calculate cumulative data
data$spbvl_cumulative <- cumsum(data$close)

# Fit the GGM model
ggm_model <- GGM(data$spbvl_cumulative, display = TRUE)

# View the model summary
summary(ggm_model)
plot(ggm_model)

# Generate future time points (e.g., for 2 years)
future_time <- (1:nrow(data)) + (1:(365 * 2))

# Predict future values
predictions <- predict(ggm_model, newx = future_time)

# Combine historical and predicted data for plotting
plot(1:nrow(data), data$spbvl_cumulative, type = "l", col = "blue",
     main = "Stock Prediction with GGM for the Next 2 Years",
     xlab = "Time", ylab = "Cumulative Close")
lines(future_time, predictions, col = "red", lty = 2)

legend("topleft", legend = c("Historical", "Predicted"), col = c("blue", "red"), lty = 1:2)

#----------------------------------------------
# Compute Pearson correlation
correlation <- cor(oni_aligned, spbvl_aligned, use = "complete.obs")
print(paste("Pearson correlation:", correlation))


# Perform correlation test, test for statistical significance
cor_test <- cor.test(oni_aligned, spbvl_aligned)
print(cor_test)

###################### ANALYSIS WITH EXTERNAL DATA (2007-2023) ##############################

# Convert the date column to Date format
data$date <- as.Date(data$date)

# Filter the data to include only records from 2007 onward
data_filtered <- data[data$date >= "2007-01-01", ]


# Create time series starting from 2007
spbvl_std_ts <- ts(data_filtered$close_std, start = c(2007, 1), frequency = 12)
#oni_std_ts <- ts(data_filtered$oni_std, start = c(2007, 1), frequency = 12)
oni_std_sst_ts <- ts(data_filtered$sst_std, start = c(2007, 1), frequency = 12)

# Align time series (ensure they have the same length)
#min_length <- min(length(oni_std_ts), length(spbvl_std_ts))
min_length <- min(length(oni_std_sst_ts), length(spbvl_std_ts))
#oni_aligned <- window(oni_std_ts, end = time(oni_std_ts)[min_length])
oni_aligned <- window(oni_std_sst_ts, end = time(oni_std_sst_ts)[min_length])
spbvl_aligned <- window(spbvl_std_ts, end = time(spbvl_std_ts)[min_length])


par(mfrow=c(1,1))
# Plot the first time series
plot(oni_std_sst_ts, type = "l", col = "blue", lwd = 2, 
     ylab = "Values", xlab = "Time", 
     main = "SPBVL and ONI Time Series")
# Add the second time series to the same plot
lines(spbvl_std_ts, col = "red", lwd = 2)

# Add a legend
legend("topright", legend = c("ONI", "SPBVL"), 
       col = c("blue", "red"), lwd = 2)


# Compute Pearson correlation
# Perform correlation test, test for statistical significance
cor_test <- cor.test(oni_aligned, spbvl_aligned)
print(cor_test)


######################### ANALYSING LAG EFFECT #################################


#Cross-Correlation Function (CCF)
#The CCF measures the correlation between two time series at different lags.

ccf_result <- ccf(oni_aligned, spbvl_aligned, lag.max = 100, plot = TRUE)
# Extract lags and their correlations
lags <- ccf_result$lag
correlations <- ccf_result$acf

# Identify significant lags
significance_threshold <- qnorm(1 - 0.05 / 2) / sqrt(length(oni_aligned))  # 95% confidence level
significant_lags <- lags[abs(correlations) > significance_threshold]
significant_values <- correlations[abs(correlations) > significance_threshold]

# Display significant lags and their correlations
data.frame(Lag = significant_lags, Correlation = significant_values)

# RESULT: No significant effect of lag. ACF is centered on 0. Analysis will
# continuing the analysis without shift.

#--------------------------GAM----------------------------------------
library(gam)
library(forecast)

tt<- (1:length(spbvl_std_ts))
#--- With splines
gm_bvl_oni <- gam(spbvl_std_ts ~ s(tt)+oni_std_sst_ts)
summary(gm_bvl_oni)
par(mfrow=c(1,2))
plot(gm_bvl_oni, se=T)
AIC(gm_bvl_oni)

#--- with loess
gm_bvl_oni_lo<- gam(spbvl_std_ts~lo(tt)+oni_std_sst_ts)
summary(gm_bvl_oni_lo)
par(mfrow=c(1,2))
plot(gm_bvl_oni_lo, se=T)
AIC(gm_bvl_oni_lo)

#--- Linear regression
l_model <- lm(spbvl_std_ts ~ oni_std_sst_ts)
summary(l_model)
AIC(l_model)

#--- Seeing the effect if oni did improve AIC rather than basic trend
gm_bvl_oni <- gam(spbvl_std_ts ~s(tt))
summary(gm_bvl_oni)
par(mfrow=c(1,2))
plot(gm_bvl_oni, se=T)
AIC(gm_bvl_oni)

# (GAM) Spline(tt)+X ----------- (GAM) Loess(tt)+X ------------- Linear Model(tt+X) ----- (Gam) Spline(tt)
# AIC: 168.22  ----------------------- 176.04 ------------------------233,18------------------192.22

##### Improving GAM with Arima model on residuals and combined model ###
gm_bvl_oni <- gam(spbvl_std_ts ~s(tt)+oni_std_sst_ts)

tsdisplay(residuals(gm_bvl_oni))

aar1<- auto.arima(residuals(gm_bvl_oni))
plot(as.numeric(spbvl_std_ts), type="l")
lines(fitted(gm_bvl_oni), col=3)
lines(fitted(aar1)+ fitted(gm_bvl_oni), col=4)


# Plot the first line (spbvl_std_ts) in the default color
plot(as.numeric(spbvl_std_ts), type = "l", col = 9, xlab = "Time", ylab = "Value", main = "GAM Fits on Index")

# Add the second line (fitted(gm_bvl_oni)) in color 3 (green)
lines(fitted(gm_bvl_oni), col = 30)

# Add the third line (fitted(aar1) + fitted(gm_bvl_oni)) in color 4 (blue)
lines(fitted(aar1) + fitted(gm_bvl_oni), col = 5)

# Optional: Add a legend to differentiate the lines
legend("topright", legend = c("spbvl_std_ts", "fitted(gm_bvl_oni)", "fitted(aar1) + fitted(gm_bvl_oni)"),
       col = c(9, 30, 5), lwd = 2)


# Plot the first line (spbvl_std_ts) in the default color
plot(as.numeric(spbvl_std_ts), type = "l", col = 9, xlab = "Time", ylab = "Value", main = "GAM Fits on Index")

# Add the second line (fitted(gm_bvl_oni)) in color 3 (green)
lines(fitted(gm_bvl_oni), col = 30)

# Add the third line (fitted(aar1) + fitted(gm_bvl_oni)) in color 4 (blue)
lines(fitted(aar1) + fitted(gm_bvl_oni), col = 5)

# Add a legend with smaller text size
legend("bottomright", 
       legend = c("spbvl_std_ts", "fitted(gm_bvl_oni)", "fitted(aar1) + fitted(gm_bvl_oni)"),
       col = c(9, 30, 5), 
       lwd = 2,
       cex = 0.9)  # Reduce text size with cex

#-------------------------- ARIMA ------------------------------------

# Load necessary library
library(forecast)

Acf(spbvl_aligned)
Pacf(spbvl_aligned)
tsdisplay(spbvl_aligned)

# Fit ARIMA(1,1,4) model, provides less AIC then auto.arima
arima_model <- arima(spbvl_aligned, order = c(1, 1, 4), xreg = oni_aligned)
summary(arima_model)
AIC(arima_model)

# Define future values of xreg (e.g., oni_aligned for forecasting period)
future_xreg <- matrix(oni_aligned, ncol = 1)# assuming oni wil be the same in future as it was

# Forecast using predict()
forecast_values <- predict(arima_model, n.ahead = length(oni_aligned), newxreg = future_xreg)

# Extract the point forecasts
predicted_values <- forecast_values$pred

# Actual data length
data_length <- length(spbvl_aligned)

# Future time points
future_time <- seq(data_length + 1, data_length + length(forecast_values$pred))

# Create the plot
plot(c(spbvl_aligned, rep(NA, length(forecast_values$pred))), type = "l", col = "black", 
     xlab = "Time", ylab = "Values", main = "SPBVL Forecast - ARIMA[1,1,4]", 
     xlim = c(1, data_length + length(forecast_values$pred)))

# Add the forecasted values
lines(future_time, forecast_values$pred, col = "red", type = "l")


# Add a legend
legend("topleft", legend = c("Actual", "Forecast"), 
       col = c("black", "red"), lty = c(1, 1, 2), bty = "n")


#---- Checking residuals
resid <- residuals(arima_model)
tsdisplay(resid)

# --------------------------AUTO-ARIAMA
arima_model <- auto.arima(spbvl_aligned, xreg = oni_aligned)
AIC(arima_model)

# Print the model summary
summary(arima_model)

# Forecast using the fitted model
forecast_values <- forecast(arima_model, xreg = oni_aligned)

# Plot the forecast
plot(forecast_values, main = "SPBVL Forecast - AUTO.ARIMA")

#---- Checking residuals
resid <- residuals(arima_model)
tsdisplay(resid)

# ----ARIMA [1,1,4] ----------- AUTO.ARIMA(3,1,1)(2,0,0)[12]
# AIC---- (-154.74) ---------------- (-149.14) -------------



#----------------------- PROPHET FORECAST ----------------------------------

install.packages("prophet")
install.packages("Metrics")
# Load necessary libraries
library(prophet)
library(Metrics)

# Prepare the data for Prophet
data_prophet <- data.frame(
  ds = as.Date(data_filtered$date),   # Date column
  y = data_filtered$close_std,        # SPBVL values (target variable)
  oni = data_filtered$sst_std        # ONI as external regressor
)

# Filter data from 2007 onward
data_prophet <- data_prophet[data_prophet$ds >= "2007-01-01", ]


library(prophet)
library(ggplot2)

# Define holiday data for COVID-19 lockdowns and the 2007-2008 financial crisis
lockdowns <- data.frame(
  holiday = c('covid_lockdown_1', 'covid_lockdown_2', 'covid_lockdown_3', 'covid_lockdown_4',
              'financial_crisis'),
  ds = as.Date(c('2020-03-21', '2020-07-09', '2021-02-13', '2021-05-28', '2008-09-15')),
  lower_window = c(0, 0, 0, 0, 0),
  upper_window = c(77, 110, 4, 13, 365) # Duration in days
)

# Split the data into 90% training and 10% testing
split_index <- floor(0.9 * nrow(data_prophet))
train_data <- data_prophet[1:split_index, ]
test_data <- data_prophet[(split_index + 1):nrow(data_prophet), ]


# Initialize the Prophet model
m <- prophet(changepoint.range = 1, 
             holidays = lockdowns,
             n.changepoints = 20,     
             seasonality.mode='multiplicative')
m <- add_seasonality(m, name = 'quarterly', period = 3, fourier.order = 4)


# Add ONI as a regressor
m <- add_regressor(m, 'oni')

# Fit the model with the training data
prophet_model_fit <- fit.prophet(m, train_data)

# Prepare the future dataframe for the test period
future <- data_prophet[, c("ds", "oni")]  # Includes both training and test periods

# Predict using the model
forecast <- predict(prophet_model_fit, future)

# Combine the forecast with actual values for comparison
results <- data.frame(
  ds = data_prophet$ds,
  actual = data_prophet$y,
  predicted = forecast$yhat
)

# Separate the training and testing data for visualization
results$set <- ifelse(results$ds <= train_data$ds[nrow(train_data)], "Training", "Testing")

ggplot(results, aes(x = ds)) +
  geom_line(aes(y = actual, color = set), size = 1) + # Different color for training and testing
  geom_line(aes(y = predicted, color = "Predicted"), size = 1, linetype = "dashed") + # Consistent prediction color
  scale_color_manual(values = c("Training" = "blue", "Testing" = "red", "Predicted" = "green")) + # Set custom colors
  labs(title = "S&P/BVL Actual vs Predicted Values-Prophet",
       x = "Date",
       y = "Value",
       color = "Legend") +
  theme_minimal()

plot(prophet_model_fit,forecast)
# Calculate MAPE for the test period
test_results <- results[results$set == "Testing", ]
mape_value <- mean(abs((test_results$actual - test_results$predicted) / test_results$actual)) * 100
print(paste("MAPE for test period:", round(mape_value, 2), "%"))


#----PROPHET AIC
# Compute residuals for the training set
train_results <- results[results$set == "Training", ]
residuals <- train_results$actual - train_results$predicted
tsdisplay(residuals)
# Compute the log-likelihood
n <- length(residuals)  # Number of observations in the training set
sigma2 <- var(residuals)  # Variance of residuals
log_likelihood <- -0.5 * n * (log(2 * pi * sigma2) + 1)

# Estimate the number of parameters
# Prophet parameters: baseline trend (k), changepoints, seasonalities, holiday effects, and regression coefficients
k <- length(prophet_model_fit$params$beta) +  # Seasonalities and additional regressors
  length(prophet_model_fit$params$delta) + # Changepoints
  1                                        # Error variance

# Calculate AIC
AIC <- 2 * k - 2 * log_likelihood

# Print the AIC value
print(paste("AIC of the Prophet model:", round(AIC, 2)))


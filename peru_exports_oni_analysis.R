# The objective of this R Script is to model the Fishmeal Export data of Peru
# and see if the ONI influences the behaviour of these exports; and can be used for modelling these exports.

# Install Packages

# install.packages("tseries")
# install.packages("forecast")
# install.packages(dynlm)
# install.packages(transfer)
# install.packages(vars)

# Import Libraries
library(tseries)
library(dplyr)
library(forecast)
library(vars)
library(dynlm)
library(rugarch)
library(mgcv)
library(stats)
library(lmtest)
library(gbm)
library(splines)



#------------------------PREPARE ONI DATA------------------------------------------------------------------------------

# Load and Prepare data for ONI data
oni_df <- read.csv("ONI_data.csv") # Read Dataframe ONI
oni_df <- oni_df[-c(1, nrow(data)),] # Drop the first and last rows

# str(oni_df$time) # To check type format of time - change from character format to date format
oni_df$time <- as.Date(oni_df$time) # Converts from Char to Date

# Since Year 1981 has only the last 3 values, its not fair to use them for doing the average.
# So, lets remove the year 1981. Same for 2023 --> Remove rows with years 1981 and 2023
oni_df <- oni_df[!(format(oni_df$time, "%Y") %in% c("1981", "2023")),]

oni_df$year <- format(oni_df$time, "%Y") # Extract Year

# Calculate average SST by year
oni_yearly_df <- oni_df %>%
  group_by(year) %>%
  summarize(
    avg_sst = mean(sst, na.rm = TRUE)
  )

# print(oni_yearly_df)

# CREATE TIME SERIES OBJECT FOR ONI
oni_values <- oni_yearly_df$avg_sst # Extract the 'avg_sst' column as the data
start_year <- min(oni_yearly_df$year) # Define the start year
oni_ts <- ts(oni_values, start = start_year, frequency = 1) # Create a time series object


# Plot the ONI time series with a black line
par(mfrow = c(1, 1)) # Display plots side-by-side
plot(oni_ts, 
     main = "Yearly Average Oceanic Niño Index (ONI)", 
     ylab = "Average SST Anomaly (°C)", 
     xlab = "Year", 
     xaxt = "n", 
     type = "l", 
     col = "black", 
     lwd = 2) 

# Customize x-axis
axis(1, at = seq(start_year, end(oni_ts)[1], by = 5), 
     labels = seq(start_year, end(oni_ts)[1], by = 5))

# Add threshold lines
abline(h = 0, col = "black", lty = 2)           # Neutral line
abline(h = 0.5, col = "darkorange", lty = 2)    # El Niño threshold
abline(h = -0.5, col = "darkblue", lty = 2)     # La Niña threshold

# Highlight El Niño and La Niña events
el_nino <- ifelse(oni_ts > 0.5, oni_ts, NA)  # El Niño values
la_nina <- ifelse(oni_ts < -0.5, oni_ts, NA) # La Niña values

points(el_nino, col = "red", pch = 19)      # Highlight El Niño events
points(la_nina, col = "skyblue", pch = 19)  # Highlight La Niña events

# Add simplified legend
legend("topleft", 
       legend = c("El Niño Event", "La Niña Event"),
       col = c("red", "skyblue"), 
       pch = c(19, 19), 
       bty = "n", 
       #inset = c(0.05, 0.05),  # Keep legend inside with padding
       #cex = 0.9
)             

# oni_ts is the time series obtained for Oceanic Nino Index.

#----------------------------PREPARE PERU EXPORT DATA----------------------------------------------------

peru_psd_df <- read.csv("peru_psd_processed.csv") # Read the Peru Data
# str(peru_psd_df)

# Remove commas from the character columns
peru_psd_df$Exports <- gsub(",", "", peru_psd_df$Exports)

# Convert the columns to numeric
peru_psd_df$Exports <- as.numeric(peru_psd_df$Exports)

# Create time series objects for each column
exports_ts <- ts(peru_psd_df$Exports, start = min(peru_psd_df$Year), frequency = 1)

# exports_ts is the time series obtained for Peru Exports.

#----------------------------CREATE STANDARDIZED MULTIVARIATE TIME SERIES DATAFRAMES----------------------------------------------------
# Since the time series datasets have different ranges, its important to standardize the data, 
# as models are sensitive to magnitude of data.

# PART I : MERGE TIME SERIES TO A SINGLE DATAFRAME

# Convert time series to data frames (with 'Year' and 'Value' columns)
df_export <- data.frame(Year = time(exports_ts), Exports = as.vector(exports_ts))  # Exports data
df_oni <- data.frame(Year = time(oni_ts), ONI = as.vector(oni_ts))  # ONI data

# Merge the data frames by "Year" to keep only common years
df <- merge(df_export, df_oni, by = "Year")

# PART II : STANDARDIZE THE MERGED DATAFRAME

# Standard scaling (mean = 0, SD = 1)
df$Exports_scaled <- scale(df$Exports)
df$ONI_scaled <- scale(df$ONI)

# Superimposed plot for both time series
par(mfrow = c(1, 1)) # Display plots side-by-side

plot(df$Year, df$Exports_scaled, type = "l", col = "blue", 
     xlab = "Year", ylab = "Standardized Values", 
     main = "Time Series of Fishmeal Exports and ONI",
     ylim = range(c(df$Exports_scaled, df$ONI_scaled)), # adjust y axis to fit both
     lwd = 2)

# Add the second time series (ONI_scaled) to the same plot
lines(df$Year, df$ONI_scaled, col = "red", lwd = 2)

# Add a legend
legend("bottomright", legend = c("Fishmeal Exports", "ONI"), 
       col = c("blue", "red"), lwd = 2)


#----------------------------CREATE TRAIN AND TEST DATAFRAMES----------------------------------------------------
# This is just to check the forecasting for the different models applied.
# Standardization is done separately on train and test dataset to make sure they don't affect each other.
# Use the mean and standard deviation from the training data to standardize the test set (to prevent data leakage).
# Save the Mean and Standard Deviation to transform the standardized data to its original form later.

# Step 1: Select only the Exports and ONI columns
df_subset <- df[, c("Year", "Exports", "ONI")]

# Step 2: Split the dataframe into train and test sets (80-20 split)
set.seed(123)  # For reproducibility
train_size <- floor(0.8 * nrow(df_subset))
train_indices <- 1:train_size
train_df <- df_subset[train_indices, ]
test_df <- df_subset[-train_indices, ]

# Step 3: Standardize the training data
train_means <- colMeans(train_df[, c("Exports", "ONI")])  # Calculate column means
train_sds <- apply(train_df[, c("Exports", "ONI")], 2, sd)  # Calculate column standard deviations

train_df_scaled <- as.data.frame(scale(train_df[, c("Exports", "ONI")], center = train_means, scale = train_sds))
train_df_scaled <- cbind(Year = train_df$Year, train_df_scaled)


# Step 4: Apply the same standardization parameters to the test data
test_df_scaled <- as.data.frame(scale(test_df[, c("Exports", "ONI")], center = train_means, scale = train_sds))
test_df_scaled <- cbind(Year = test_df$Year, test_df_scaled)


# Step 5: Print the resulting dataframes
print("Training Data (Scaled):")
print(head(train_df_scaled))

print("Testing Data (Scaled):")
print(head(test_df_scaled))

# I basically try out both, model and forecast on the entire dataset. If its a good model, then
# I try to model it on the train and evaluate on the test dataset.

# Convert train and test data to time series
train_ts_scaled <- ts(train_df_scaled$Exports, start = min(train_df_scaled$Year), frequency = 1)
test_ts_scaled <- ts(test_df_scaled$Exports, start = min(test_df_scaled$Year), frequency = 1)

print(train_ts_scaled)

#----------------------------ANALYSIS OF EXPORT TIME SERIES----------------------------------------------------
# Since I don't have the ONI data before 1982, I think it is fair to model the exports from the same year as ONI.
# Also, the standardized (scaled) data is used for the analysis.

# Create the time series plot for Fish Export
par(mfrow = c(1, 1)) # Display plots side-by-side
plot(df$Year,df$Exports_scaled, 
     main = "Peru Annual Fishmeal Export Data", 
     ylab = "Standardized Value", 
     xlab = "Year", 
     type = "l",  # Line plot without points
     col = "blue", 
     lwd = 2)     # Line thickness
grid(nx = NULL, ny = NULL, col = "gray", lty = "dotted", lwd = 0.5) # Add grid lines for better readability
abline(h = mean(df$Exports_scaled), col = "darkgreen", lty = 2, lwd = 1.5) # Add a horizontal line indicating the mean export level


# Insights
# 1. Trend : Upward trend from 1980s to late 1990s, followed by a fluctuating slow decline.
# 2. Seasonality : No seasonality as such, but periodic peaks and troughs suggest cyclic behaviour.
# 3. Volatility : Around late 1990s, the sharp negative spike, followed by a peak suggests external factors influencing the exports.
# 4. Non Stationary due to the visible trend.


# AUTOCORRELATION CHECK USING ACF and PACF

# Significant autocorrelations at lag 1 or 2 in the ACF plot, 
# indicate that past values have a strong influence on future values, 
# which is characteristic of an AR (AutoRegressive) process.

# The PACF plot helps determine the number of autoregressive terms (AR) 
# and moving average terms (MA) for an ARIMA model.

par(mfrow = c(1, 2)) # Display plots side-by-side
acf(df$Exports_scaled, main = "ACF of Peru Fish Exports")
pacf(df$Exports_scaled, main = "PACF of Peru Fish Exports")

# Insights
# ACF - Slow decay, non-stationary due to trend. First few lags are highly significant i.e. strong autocorrelation.
# PACF - Significant spike at lag 1, possibly at lag 3 too, rest are relatively smaller lags.
# Suggests first differencing and Autoregressive (1) and (3).


# TESTING FOR NON-STATIONARITY

# Null Hypothesis (H0): The time series is non-stationary.
# Alternative Hypothesis (H1): The time series is stationary.
# If the p-value > 0.05, you fail to reject the null hypothesis, meaning the series is non-stationary.
# If the p-value < 0.05, you reject the null hypothesis, meaning the series is stationary.

adf_test_result <- adf.test(df$Exports_scaled)
print(adf_test_result)

# Interpretation of the ADF Test Result:
# p-value = 0.48 --> Time Series is non-stationary. Implies we have to make the time series stationary.


# APPLYING FIRST DIFFERENCING TO ACHIEVE STATIONARITY
# Note : I cannot proceed with the dataframe as first differencing leads to NA value in the first row.

df$Exports_scaled_diff <- c(NA, diff(df$Exports_scaled)) # JUST KEEPING IT.

# First difference of non-stationary time series
ts_Exports_scaled_diff <- diff(df$Exports_scaled)

# Checking ACF and PACF for the differenced series
par(mfrow = c(1, 2)) # Display plots side-by-side
acf(ts_Exports_scaled_diff, main = "ACF of Differenced Peru Exports")
pacf(ts_Exports_scaled_diff, main = "PACF of Differenced Peru Exports")

# Insights
# ACF has sharp drop after lag 1, insignificant autocorrelations at later lags -> Stationarity achieved.
# PACF shows minor spike at lag 2, so we can try AR(2) later perhaps.

# Verifying Stationarity
adf_test_result <- adf.test(ts_Exports_scaled_diff)
print(adf_test_result)
# p-value = 0.01 --> Series is stationary now!

#----------------------------ANALYSIS OF ONI TIME SERIES----------------------------------------------------

# Create the time series plot for ONI
par(mfrow = c(1, 1)) # Display plots side-by-side
plot(df$Year,df$ONI_scaled, 
     main = "ONI Time Series", 
     ylab = "Value", 
     xlab = "Year", 
     type = "l",  # Line plot without points
     col = "blue", 
     lwd = 2)     # Line thickness
grid(nx = NULL, ny = NULL, col = "gray", lty = "dotted", lwd = 0.5) # Add grid lines for better readability
abline(h = mean(df$ONI_scaled), col = "darkgreen", lty = 2, lwd = 1.5) # Add a horizontal line indicating the mean export level

# Insights
# cyclical pattern with peaks and troughs recurring every few years i.e. some periodicity, NOT seasonality.
# No trend or annual seasonality.

# Determining the Periodicity of the Cycle (Spectral analysis using Fourier Transform)

# Perform Fourier Transform
fft_result <- fft(df$ONI_scaled)

# Compute frequencies
n <- length(df$ONI_scaled)  # Number of observations
freq <- (0:(n-1)) / n       # Frequency for each component

# Compute power spectrum
power_spectrum <- Mod(fft_result)^2 / n

# Identify dominant frequency
dominant_freq_index <- which.max(power_spectrum[2:(n/2)]) + 1  # Skip the first (DC component)
dominant_frequency <- freq[dominant_freq_index]

# Convert frequency to periodicity (in years)
dominant_period <- 1 / dominant_frequency

# Print results
cat("Dominant Period (in years):", dominant_period, "\n")

# Plot Power Spectrum
plot(freq[1:(n/2)], power_spectrum[1:(n/2)], type = "b", col = "blue",
     xlab = "Frequency", ylab = "Power Spectrum",
     main = "Power Spectrum of ONI Time Series")
abline(v = dominant_frequency, col = "red", lty = 2)
text(dominant_frequency, max(power_spectrum), labels = paste0("Period = ", round(dominant_period, 2)), pos = 4, col = "red")

#----------------------------CORRELATION BETWEEN ONI-SST AND PERU EXPORT TIME SERIES----------------------------------------------------

# PERU EXPORT TIME SERIES - is non - stationary. So we will use the differenced time series
# ONI - Stationary, so we will use it like that.
# Basically we are checking : how the changes in exports are related to the changes in the ONI series.
# Still using scaled values

# Find common years
common_years <- intersect(time(df$ONI_scaled), time(ts_Exports_scaled_diff))

# Subset the series to the common years
oni_aligned <- window(df$ONI_scaled, start = min(common_years), end = max(common_years))
exports_aligned <- window(ts_Exports_scaled_diff, start = min(common_years), end = max(common_years))

# Confirm the lengths match
length(oni_aligned)
length(exports_aligned)

# correlations
cor(oni_aligned, exports_aligned, method = "pearson") # correlation is -0.44

cor(oni_aligned, exports_aligned, method = "spearman") # correlation is -0.50

#----------------------------APPLYING ARIMA MODELS ON EXPORT DATA----------------------------------------------------

# ARIMA Models (p, d, q) to be considered : 

# 1. ARIMA(1,1,0): First-order autoregressive model with differencing.
# 2. ARIMA(0,1,1): First-order moving average model with differencing.
# 3. ARIMA(1,1,1): A combination of AR(1) and MA(1).
# 4. ARIMA(2,1,0): AR(2) with differencing.
# 5. ARIMA(2,1,1): A combination of AR(2) and MA(1).

# I will apply ARIMA on the original series, and put differencing as 1. i.e. d=1
# Note : ARIMA requires time series data, so convert df column to time series

# ts_exports_scaled <- ts(df$Exports_scaled, start = min(df$Year), frequency = 1)  # Frequency = 1 for annual data
# str(ts_exports_scaled)
ts_exports_scaled <- ts(as.numeric(df$Exports_scaled), start = min(df$Year), frequency = 1)


# Fit ARIMA(1,1,0) model
arima_110 <- Arima(ts_exports_scaled, order = c(1, 1, 0))
summary(arima_110)

# Fit ARIMA(0,1,1) model
arima_011 <- Arima(ts_exports_scaled, order = c(0, 1, 1))
summary(arima_011)

# Fit ARIMA(1,1,1) model
arima_111 <- Arima(ts_exports_scaled, order = c(1, 1, 1))
summary(arima_111)

# Fit ARIMA(2,1,0) model
arima_210 <- Arima(ts_exports_scaled, order = c(2, 1, 0))
summary(arima_210)

# Fit ARIMA(2,1,1) model
arima_211 <- Arima(ts_exports_scaled, order = c(2, 1, 1))
summary(arima_211)

# Fit ARIMA (1,1, 3) model
arima_113 <- Arima(ts_exports_scaled, order = c(1, 1, 3))
summary(arima_113)

# Trying out autoarima
autoarima_model <- auto.arima(ts_exports_scaled)
summary(autoarima) # It gave ARIMA (1, 0, 0) with AIC 99.99, BIC = 103.42, MAPE = 241

# Compare AIC values
AIC(arima_110, arima_011, arima_111, arima_210, arima_211, arima_113, autoarima_model)
# ARIMA_210 has the lowest AIC value of 92.98; rest are also quite comparable in the range (93-101)

# Compare BIC values
BIC(arima_110, arima_011, arima_111, arima_210, arima_211, arima_113, autoarima)
# ARIMA_210 has the lowest BIC value of 98.04; rest are also close by in the range (99-104)


# Check residuals for the best model (i.e. arima_210)
checkresiduals(arima_210)

# ACF is within bounds, residuals are fluctuating around mean 0, histogram centered around 0
# But there is 1 event near 1997 which show higher variance (corresponds to El Nino event)
# So maybe, not everything has been modelled.

# confirm if the residuals are actually white noise
Box.test(residuals(arima_210), lag = 10, type = "Ljung-Box") # p-value = 0.82 > 0.05, implies it is white noise.

#----------------------------FORECASTING USING ARIMA (2, 1, 0) ----------------------------------------------------

# Forecast the next few periods using best model
par(mfrow = c(1, 1))
forecast_arima_210 <- forecast(arima_210, h = 10)
plot(forecast_arima_210, main = "Forecast of Exports with ARIMA (2, 1, 0)") # Plot the forecast


# Try with train and test set to see predictive error.
# train_ts_scaled and test_ts_scaled

# Fit ARIMA model on training data
arima_model <- Arima(train_ts_scaled, order = c(2, 1, 0))
summary(arima_model)

# Forecast for the length of the test set
forecasted_values <- forecast(arima_model, h = length(test_ts_scaled))
predicted_values <- forecasted_values$mean

# Calculate MAPE
mape <- mean(abs((test_ts_scaled - predicted_values) / test_ts_scaled)) * 100

# Calculate RMSE
rmse <- sqrt(mean((test_ts_scaled - predicted_values)^2))

# Print error metrics
cat("MAPE:", mape, "%\n")
cat("RMSE:", rmse, "\n")

plot(forecasted_values, main = "ARIMA Forecast vs Actual", xlab = "Year", ylab = "Exports (Scaled)")
lines(test_ts_scaled, col = "red", lty = 2)  # Overlay the actual test data
legend("topleft", legend = c("Forecast", "Actual (Test)"), col = c("blue", "red"), lty = c(1, 2), bty = "n")

# Its not a good forecasting model.

#----------------------------FORECASTING USING ARIMA (2, 1, 0) WITH ONI AS EXTERNAL VARIABLE ----------------------------------------------------
# Dynamic Regression Model

ts_oni_scaled <- ts(df$ONI_scaled, start = min(df$Year), frequency = 1)  # Frequency = 1 for annual data

# Fit ARIMAX with ONI as an external regressor
arimax_model <- Arima(ts_exports_scaled, order = c(2, 1, 0), xreg = ts_oni_scaled)
summary(arimax_model)
forecast_arimax <- forecast(arimax_model, xreg = ts_oni_scaled, h=10)
par(mfrow = c(1, 1))
plot(forecast_arimax, main='Forecast from Regression with ARIMAX(2, 1, 0)')
checkresiduals(arimax_model) 
# AIC = 85, bic = 92, MAPE = 234

# confirm if the residuals are actually white noise
Box.test(residuals(arimax_model), lag = 10, type = "Ljung-Box") # p-value = 0.98 > 0.05, implies it is white noise.

#----------------------------OTHER MODELS ----------------------------------------------------

# Residual Analysis for Models: ACF on residuals should not throw any significant lags.

# Linear Regression with External Variable ONI.
# Assumes linear relationship b/w variables

lm_model <- lm(Exports_scaled ~ ONI_scaled, data = df)
summary(lm_model)
checkresiduals(lm_model)
acf(residuals(lm_model)) # residuals should not have significant autocorrelation, but they do.

# Exponential Smoothing (ETS)
ets_model <- ets(ts_exports_scaled)
summary(ets_model) # AIC value = 135, BIC = 140
forecast_ets <- forecast(ets_model, h = 12)
plot(forecast_ets)
checkresiduals(ets_model) 
acf(residuals(ets_model)). # No significant autocorrelations, so better model than linear regression --> non-linear relationship


#----------------------------NON PARAMETRIC REGRESSION MODELS ----------------------------------------------------

# 1. Loess is a non-parametric regression method that fits local linear or 
# polynomial regressions to small subsets of data.
# Best for local smoothing and capturing small-scale fluctuations.

# Ensure the data is in a data frame
data <- data.frame(
  time = 1:length(df$Exports_scaled),
  Exports_scaled = df$Exports_scaled,
  ONI_scaled = df$ONI_scaled
)

# Fit a LOESS model with Exports as the response variable
loess_model <- loess(Exports_scaled ~ time + ONI_scaled, data = data, span = 0.3)
summary(loess_model)


# Predict values using the fitted LOESS model
loess_predictions <- predict(loess_model, newdata = data)

# Add predictions to the data frame for plotting
data$Predicted_Exports <- loess_predictions

# Plot the actual data
plot(
  data$time, data$Exports_scaled, 
  type = "l", col = "blue", lwd = 2, 
  ylab = "Exports (Scaled)", xlab = "Time", 
  main = "LOESS Fit with ONI as an External Variable"
)

# Add the predicted values to the plot
lines(data$time, data$Predicted_Exports, col = "red", lwd = 2, lty = 2)

# Add a legend
legend(
  "topright", legend = c("Actual Exports", "Predicted Exports (LOESS)"),
  col = c("blue", "red"), lty = c(1, 2), lwd = 2
)

# Calculate evaluation metrics
mse <- mean((data$Exports_scaled - data$Predicted_Exports)^2)
mae <- mean(abs(data$Exports_scaled - data$Predicted_Exports))

print(c(MSE = mse, MAE = mae)). # 0.07, 0.19

checkresiduals(loess_model)



# 2. Generalized Additive Model (GAM)
# GAM supports nonlinear relationships and can include oni_ts as an external variable.

# Create a data frame with time, Exports, and ONI
data <- data.frame(
  time = 1:length(df$Exports_scaled),
  Exports_scaled = df$Exports_scaled,
  ONI_scaled = df$ONI_scaled
)

# Fit GAM model: Exports as a function of time and ONI
gam_model <- gam(Exports_scaled ~ s(time) + s(ONI_scaled), data = data)

# Summary of the model
summary(gam_model)

# Plot diagnostic plots for the GAM model
par(mfrow = c(2, 2))
plot(gam_model)

# Generate predicted values for the observed data
data$fitted_values <- predict(gam_model, newdata = data)

par(mfrow = c(1, 1))
# Plot original data points
plot(
  data$time, data$Exports_scaled,
  main = "GAM Fitted Curve",
  xlab = "Time", ylab = "Exports (Scaled)",
  pch = 16, col = "blue"  # Style the points
)

# Add the fitted curve
lines(data$time, data$fitted_values, col = "red", lwd = 2)


# Sort data by ONI_scaled for a smooth curve
data <- data[order(data$ONI_scaled), ]

# Plot ONI effect
plot(
  data$ONI_scaled, data$Exports_scaled,
  main = "GAM Effect of ONI",
  xlab = "ONI (Scaled)", ylab = "Exports (Scaled)",
  pch = 16, col = "blue"
)

# Add the fitted curve
lines(data$ONI_scaled, data$fitted_values, col = "red", lwd = 2)

# Calculate residuals
data$residuals <- data$Exports_scaled - data$fitted_values

# Mean Squared Error (MSE)
mse <- mean(data$residuals^2)
cat("Mean Squared Error (MSE):", mse, "\n")  # 0.18

# Mean Absolute Error (MAE)
mae <- mean(abs(data$residuals))
cat("Mean Absolute Error (MAE):", mae, "\n") #0.34












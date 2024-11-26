# Load the data
oni <- read.csv("ONI_data.csv")
str(oni)

# Turn the important columns into variables
value<-oni&ONI

# Plot some simple plots
plot(value, type = "b", pch = 19, col = "blue", lty = 2, cex = 0.5)
plot(cumsum(value), type = "b", pch = 19, col = "blue", lty = 2, cex = 0.5)

# See that the autocorrelation is huge
Acf(value)
Pacf(value)
# In fact, there is a huge spike at lag 1 and 2 in PACF, and nothing else. It may be an 
# ARMA(1,d,0) or ARMA(2,d,0).

# See as well the correlation in the cumulative function, it's even bigger
Acf(cumsum(value))
Pacf(cumsum(value))

# Fit an arima model
auto.a<- auto.arima(ts(value))
checkresiduals(auto.a)
# The best model according to autoplot is the ARIMA(4,0,5)

# Might as well fit an arima model for the cumulative function
auto.a_cum<- auto.arima(ts(cumsum(value)))
checkresiduals(auto.a_cum)
# In this case, the the best model is ARIMA(3,1,5)

# Forecast the future
forecast_oni <- forecast(auto.a, h=24)
autoplot(forecast_oni)






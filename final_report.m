%load data
loaded = readtable('TRAINING/train2.csv');
t = loaded.Time;
y = loaded.pH;

%plot data
fh = figure;
plot(t/60,y)
xlabel('Time (minute)')
ylabel('pH')
xlim([0 60])
grid on

saveas(fh, 'illustration_data.png')

meanvalue = mean(y);
%variance
variance = var(y);
%standard deviation
standarddev = std(y);
%autocorrelation & parcorr

%from the plot, it seems that the series is not stationary: we can notice
%some relevant trends and changing levels, but we want to be sure

%PRE-PROCESSING: CHECK STATIONARITY
% Stationarity means that the statistical properties of a a time series (or
%rather the process generating it) do not change over time.

% Stationarity is important because many useful analytical tools and statistical 
%tests and models rely on it.

%a way to assess stationarity is autocorr:
%Autocorrelation is the correlation of a signal with a delayed copy — or a lag — 
%of itself as a function of the delay. When plotting the value of the ACF for
%increasing lags (a plot called a correlogram), the values tend to degrade to zero
%quickly for stationary time series), while for non-stationary data the
%degradation will happen more slowly.
fh = figure;
subplot(2,1,1)
autocorr(y)
title('ACF of the pH')
subplot(2,1,2)
parcorr(y)
title('PACF of the pH')

%Another, more rigorous approach, to detecting stationarity in time series data
%is using statistical tests developed to detect specific types of stationarity,
%namely those brought about by simple parametric models of the generating stochastic process
adftest(y) %if 0 non-stat, if 1 stat
kpsstest(y)

%A way to make a time series stationary is to take its difference: for each
% value in our time series we subtract the previous value. This will give
% us a NaN value at first place because there is no previous value: to
% avoid this I will pad the new vector with a 0
Y = diff(y);
Y = [0;Y]


fh = figure;
plot(t/60,Y)
xlabel('Time (minute)')
ylabel('pH')
xlim([0 60])
grid on

saveas(fh, 'illustration_stationarity.png')
%check with tests:
adftest(Y)
figure
autocorr(Y)
title('ACF of the stationary pH')
%time series analysis
%mean
smeanvalue = mean(Y);
%variance
svariance = var(Y);
%standard deviation
sstandarddev = std(Y);
%autocorrelation & parcorr
figure
subplot(2,1,1)
autocorr(Y)
title('ACF of the pH')
subplot(2,1,2)
parcorr(Y)
title('PACF of the pH')

%plot the spectrum
figure
my_spectrum = periodogram(Y);
plot(my_spectrum); %no characteristic peaks

%AR
ARMdl = ar(Y, 3)
pred = compare(Y, ARMdl, 1); %one step prediction
figure
plot(t/60, [Y pred])
legend("True", "Predicted")
grid on
figure
err = Y - pred;
histogram(err, 50);

%ARMA
na = 3;
nc = 1;
ARMAMdl = armax(Y,[na nc])
pred = compare(Y, ARMAMdl, 1); %one step prediction
figure
plot(t/60, [Y pred])
legend("True", "Predicted")
grid on
figure
err = Y - pred;
histogram(err, 50);

%ARMAX (same of ARMA because I have no inputs)
nb = []; %armax expects a matrix for nb and nk
nk = [];
Ts = 1;
data = iddata(Y,[],Ts) 
ARMAXMdl = armax(data,[na nk nc nk])
pred = compare(data, ARMAXMdl, 1); %one step prediction
figure
plot(t/60, [data.y pred.y])
legend("True", "Predicted")
grid on
figure
err = data.y - pred.y;
histogram(err, 50);

%SARIMA
SARIMAMdl = arima('Constant',NaN,'ARLags',1:3,'D',1,'MALags',1:2,'SARLags',[],'Seasonality',0,'SMALags',[],'Distribution','Gaussian');
SARIMAEst = estimate(SARIMAMdl,y)
res = infer(SARIMAEst,y);
figure
subplot(2,2,1)
plot(res./sqrt(SARIMAEst.Variance))
title('Standardized Residuals') 
subplot(2,2,2)
qqplot(res)
subplot(2,2,3)
autocorr(res)
subplot(2,2,4)
parcorr(res)
hvec = findall(gcf,'Type','axes'); set(hvec,'TitleFontSizeMultiplier',0.8,'LabelFontSizeMultiplier',0.8);
% Generate forecast
[yF,yMSE] = forecast(SARIMAEst,100,'Y0',y);
figure
plot(yF)

%NAR
%MADE WITH NTSTOOL
%myNeuralNetworkFunction(...)

%GARCH
% === Stimiamo un modello GARCH
%definiamo la struttura del modello GARCH
ngarchLags=2;   % la parte garch rappresenta le regressioni della sig^2
narchLags=1;    % la parte arch rappresenta le regressioni della eps^2
garchMdl = garch('Offset',0,'GARCHLags',ngarchLags,'ARCHLags',narchLags,'Distribution','Gaussian');
GARCHEst = estimate(garchMdl,y);
% === Simuliamo 100 paths del modello
numObs = numel(y); % Sample size (T)
numPaths = 100;     % Number of paths to simulate
rng(1);             % For reproducibility
[VSim,YSim] = simulate(GARCHEst,numObs,'NumPaths',numPaths); % VSim è la varianza, YSim gli eps
%VSim and %YSim are numObs  * numPaths (so 63900*100) matrices. Rows
%correspond to a sample per period and columns correspond to a simulated
%path.

%Plot the average and the 97.5% and 2.5% percentiles of the simulated
%paths.
VSimBar = mean(VSim, 2);
VSimCI = quantile(VSim, [0.025 0.975], 2);
YSimBar = mean(YSim, 2);
YSimCI = quantile(YSim, [0.025 0.975], 2);

figure;
subplot(2,1,1);
h1=plot(t, VSim, 'Color', 0.8*ones(1,3));
hold on;
h2=plot(t, VSimBar, 'k--', 'LineWidth', 2);
h3=plot(t, VSimCI, 'r--', 'LineWidth', 2);
hold off;
title('Simulated Conditional Variances');
ylabel('Cond. var.');
xlabel('Time');

subplot(2,1,2);
h1=plot(t, YSim, 'Color', 0.8*ones(1,3));
hold on;
h2=plot(t, YSimBar, 'k--', 'LineWidth', 2);
h3=plot(t, YSimCI, 'r--', 'LineWidth', 2);
hold off;
title('Simulated Nominal Returns');
ylabel('Nominal return (%)');
xlabel('Time');
legend([h1(1) h2 h3(1)], {'Simulated path' 'Mean' 'Confidence bounds'}, 'FontSize', 7, 'Location', 'NorthWest');

numPeriods = 10;
vF = forecast(GARCHEst, numPeriods, YSim);

v = infer(GARCHEst, YSim);
figure;
plot(t, v, 'k:', 'LineWidth',2);
hold on;
plot(t(end):t(end)+10,[v(end);vF],'r','LineWidht',2);
title('Forecasted Conditional Variances of Nominal Returns');
ylabel('Conditional variances');
xlabels('Time');


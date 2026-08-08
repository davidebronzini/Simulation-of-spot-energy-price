import pandas as pd
import numpy as np
import openpyxl
import matplotlib.pyplot as plt
from ISLP.models import (ModelSpec as MS, summarize, poly)
import statsmodels.api as sm
from scipy.stats import norm
from scipy.optimize import minimize
from pandas.tseries.offsets import MonthEnd
#pre-processing
raw=pd.read_excel("DAILY_PUN.xlsx") #daily pun from 2020 to 2025
print(raw)
raw.columns=["DATE","PRICE"]
raw["DATE"]=pd.to_datetime(raw["DATE"],format="%d/%m/%Y")
raw["PRICE"]=pd.to_numeric(raw["PRICE"].str.replace(",","."))
print(raw)
plt.plot(raw["DATE"],raw["PRICE"])
plt.show()
# 1 time in annual format
t=raw["DATE"].dt.dayofyear/365.25
print(t)
#2 logprice
lnP=np.log(raw["PRICE"])
print(lnP)
df = pd.DataFrame({"Date":raw["DATE"],
    "t": t,
    "lnP": lnP
}).dropna().reset_index(drop=True)

print(df)


df["x1"] = np.sin(2 * np.pi * df["t"])
df["x2"] = np.cos(2 * np.pi * df["t"])
df["x3"] = np.sin(4 * np.pi * df["t"])
df["x4"] = np.cos(4 * np.pi * df["t"])

predictors = MS(["x1", "x2", "x3", "x4"])
print(predictors)
X=predictors.fit_transform(df)
y=df["lnP"]
model=sm.OLS(y,X)
model=model.fit()
print(model.params)
#4 compute seasonality manually or with .predict() method
season_params=model.params.values
theta=X.values @ season_params
print(theta)
df["Theta"]=theta
print(df)
plt.plot(df["Date"], df["Theta"], label="Seasonality")
plt.plot(df["Date"], df["lnP"], label="Log price")
plt.title("Log price vs seasonality")
plt.xlabel("t")
plt.legend()
plt.show()
# Deseasonalized series
df["lnP-season"] = df["lnP"] - df["Theta"]

# Create figure with 2 subplots
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12,8))

# First subplot: original vs seasonality
axes[0].plot(df["Date"], df["lnP"], label="Log price")
axes[0].plot(df["Date"], df["Theta"], label="Seasonality")
axes[0].set_title("Log price vs seasonality")
axes[0].legend()

# Second subplot: deseasonalized series
axes[1].plot(df["Date"], df["lnP-season"], label="Deseasonalized log price")
axes[1].set_title("Deseasonalized log price")
axes[1].legend()

plt.tight_layout()
plt.show()

#5 discretization 
dt=1/365.25
Xt=df["lnP-season"].iloc[1:].to_numpy()
Xt_prev=df["lnP-season"].iloc[:-1].to_numpy()

#MLE (comparison model paper (additive jump vs my model multiplicative jump)
def minus_loglike(betas,Xt,Xt_prev,dt):
    mu,k,sigma,mu_j,sigma_j,lambd=betas
    m=Xt_prev+(mu-k*Xt_prev)*dt
    var=sigma**2*dt
    phi_0=norm.pdf(Xt,loc=m,scale=np.sqrt(var))
    phi_1 = norm.pdf(Xt,loc=m+Xt_prev*mu_j,scale=np.sqrt(var+Xt_prev**2*sigma_j**2))
    density=(1-lambd*dt)*phi_0+lambd*dt*phi_1

    loglike=np.sum(np.log(density))

    return -loglike



betas0=[
    0.1,    # mu
    1,      # k
    0.2,    # sigma
    0,      # mu_j
    0.1,    # sigma_j
    5       # lambda
]
print(minus_loglike(
    betas0,
    Xt,
    Xt_prev,
    dt
))

boun=[
(None,None),        # mu
(0,None),     # k
(0,None),     # sigma
(None,None),     # mu_j
(0,None),     # sigma_j
(1e-6,300)       # lambda
]

opt=minimize(minus_loglike,betas0, args=(Xt, Xt_prev, dt),method="L-BFGS-B",bounds=boun)
print("MY model:",opt)

def minus_loglike2(betas,Xt,Xt_prev,dt):
    mu,k,sigma,mu_j,sigma_j,lambd=betas
    m=Xt_prev+(mu-k*Xt_prev)*dt
    var=sigma**2*dt
    phi_0=norm.pdf(Xt,loc=m,scale=np.sqrt(var))
    phi_1 = norm.pdf(Xt,loc=m+mu_j,scale=np.sqrt(var+sigma_j**2))
    density=(1-lambd*dt)*phi_0+lambd*dt*phi_1

    loglike=np.sum(np.log(density))

    return -loglike



opt2=minimize(minus_loglike2,betas0, args=(Xt, Xt_prev, dt),method="L-BFGS-B",bounds=boun)

print("PAPER model:",opt2)



logL = -opt.fun
logL2 = -opt2.fun
n = len(Xt)

p = 6

AIC = 2*p - 2*logL
BIC = p*np.log(n) - 2*logL

print(f"Log-likelihood = {logL:.3f}")
print(f"AIC = {AIC:.3f}")
print(f"BIC = {BIC:.3f}")

AIC2 = 2*p - 2*logL2
BIC2 = p*np.log(n) - 2*logL2

print(f"Log-likelihood = {logL2:.3f}")
print(f"AIC = {AIC2:.3f}")
print(f"BIC = {BIC2:.3f}")

#we choose paper model
mu=opt2.x[0]
k=opt2.x[1]
sigma=opt2.x[2]
mu_j=opt2.x[3]
sigma_j=opt2.x[4]
lambd=opt2.x[5]

# 6 load futures data and pre process
raw_fut=pd.read_excel("FUTURES.xlsx",sheet_name="Sheet1")
print(raw_fut)
raw_fut=raw_fut.drop([0,1],axis=0).reset_index(drop=True)
print(raw_fut)
futures=pd.DataFrame(raw_fut)
futures["DATE"]=pd.to_datetime(futures["DATE"])
futures=futures.set_index("DATE",drop=False)
#set the date for evaltuation for forward curve
fwdcurve_date=df["Date"].iloc[-1]# qui bisognera inserire gestione caso quotazione pun non combacia con quotazione future
print(fwdcurve_date)
#set the expiration of the futures
future_df=futures.melt(id_vars="DATE", var_name="EXPIRATION", value_name="F")
print(future_df)

mesi = {
    "Gen": "Jan",
    "Feb": "Feb",
    "Mar": "Mar",
    "Apr": "Apr",
    "Mag": "May",
    "Giu": "Jun",
    "Lug": "Jul",
    "Ago": "Aug",
    "Set": "Sep",
    "Ott": "Oct",
    "Nov": "Nov",
    "Dic": "Dec",
}
future_df["PRODUCT"]=future_df["EXPIRATION"]
future_df["EXPIRATION"]= future_df["EXPIRATION"].replace(mesi, regex=True)

future_df["EXPIRATION"] = pd.to_datetime(future_df["EXPIRATION"], format="%b-%y")

future_df["EXPIRATION"]=future_df["EXPIRATION"]- pd.offsets.BDay(2)
print(future_df)
future_df = future_df[future_df["DATE"] == fwdcurve_date]

print(future_df)
#compute exaxt T to decide the simulation end

# prendo solo futures disponibili
available = future_df.dropna(subset=["F"])

# ultimo contratto disponibile
last_contract = available["EXPIRATION"].max()

# ultimo giorno del mese del contratto
simulation_end = last_contract + MonthEnd(0)

# numero periodi simulazione
T = (simulation_end - fwdcurve_date).days + 1

print("Forward curve date:", fwdcurve_date)
print("Simulation end:", simulation_end)
print("T =", T)







#7 montecarlo simulation pun
np.random.seed(50)
T=T
trials=10000
e=np.random.normal(size=[T,trials])
N=np.random.poisson(lambd*dt,size=[T,trials])
J=np.random.normal(mu_j,sigma_j,size=[T,trials])
jump = (N > 0) * J
X_sim=np.zeros((T,trials))
X_sim[0,:]=Xt[-1]
for i in range(1,T):
    X_sim[i,:]=X_sim[i-1,:]+(mu-k*X_sim[i-1,:])*dt+sigma*np.sqrt(dt)*e[i,:]+jump[i,:]
#plotting simulation
timeline=np.arange(T)*dt
plt.plot(timeline,X_sim[:, :30])
plt.show()
#computing dates of the simulation
start_sim=df["Date"].iloc[-1] + pd.Timedelta(days=0)
date_sim=pd.date_range(start_sim,periods=T,freq="D")  

#8 compututing seasonality of simulation part
time_sim = (date_sim - df["Date"].iloc[0]).days.astype(float) / 365.25
df_sim = pd.DataFrame({"Date": date_sim,"t": time_sim})
df_sim["x1"] = np.sin(2 * np.pi * df_sim["t"])
df_sim["x2"] = np.cos(2 * np.pi * df_sim["t"])
df_sim["x3"] = np.sin(4 * np.pi * df_sim["t"])
df_sim["x4"] = np.cos(4 * np.pi * df_sim["t"])

X2=predictors.transform(df_sim)
theta_sim=X2.values @ season_params
#log price simulation
logP_sim=theta_sim[:, None] + X_sim
#price simulation
P_sim=np.exp(logP_sim)
#plotting the actual+simulation price and seasonality
df_sim["SIM_LOG_P"]=logP_sim[:,0] # chosen trajectory
df_sim["SIM_theta"]=pd.Series(theta_sim)
plt.plot(df["Date"], df["lnP"], label="Log price")
plt.plot(df["Date"], df["Theta"], label="Seasonality")
plt.plot(df_sim["Date"],df_sim["SIM_LOG_P"], label="SIMULATION PUN")
plt.plot(df_sim["Date"],df_sim["SIM_theta"], label="SIMULATION season")
plt.title("PUN+SIMULATION")
plt.legend()
plt.tight_layout()
plt.show()


#9 expected futures price under real world P
E_FUT = np.mean(P_sim, axis=1)


#10 set hp: F= for all day of the month
F_daily = np.zeros(date_sim.shape)
for i in range(0,T):

    mask = (
        (future_df["EXPIRATION"].dt.year == date_sim[i].year) &
        (future_df["EXPIRATION"].dt.month == date_sim[i].month)
    )
    
    sel = future_df.loc[mask, "F"]

    if not sel.empty:
        F_daily[i] = sel.iloc[0]

    else:
        
        F_daily[i] = np.nan

print(F_daily)


#11comput the market price of risk m

t0 = (fwdcurve_date - df["Date"].iloc[0]).days / 365.25
tz=time_sim-t0
#we nned to thrat nan of F_daily
new_mask=~np.isnan(F_daily)
F_daily_cl=F_daily[new_mask]
E_FUT_cl=E_FUT[new_mask]
tz_cl=tz[new_mask]


b=-np.log(F_daily_cl[1:]/E_FUT_cl[1:])/(sigma*np.exp(-k*tz_cl[1:]))

delta=(1/ k)*(np.exp(k*tz_cl[1:]))-(np.exp(k*tz_cl[0:-1]))

A=np.tril(np.tile(delta,(delta.size, 1)))

P=np.diag(1/np.diag(A))
b=P @ b #matrix pprodutc
A=P @ A
riskpremium=np.linalg.solve(A,b)
# check if there are errors
print(len(F_daily_cl))
print(len(E_FUT_cl))
print(len(tz_cl))
print(A.shape)
print(b.shape)
print(riskpremium)

plt.plot(date_sim[1:],riskpremium)
plt.show()

#12 simulation under risk neutral measure
#manage problem T is different from lenght of risk premium because many F_daily are cut off
riskpremium_full = np.empty(T)
riskpremium_full[:len(riskpremium)] = riskpremium
riskpremium_full[len(riskpremium):] = riskpremium[-1]
np.random.seed(50)
trials=10000
e=np.random.normal(size=[T,trials])
N=np.random.poisson(lambd*dt,size=[T,trials])
J=np.random.normal(mu_j,sigma_j,size=[T,trials])
jump = (N > 0) * J
X_sim=np.zeros((T,trials))
X_sim[0,:]=Xt[-1]
for i in range(1,T):
    X_sim[i,:]=X_sim[i-1,:]+(mu-sigma*riskpremium_full[i-1] -k*X_sim[i-1,:])*dt+sigma*np.sqrt(dt)*e[i,:]+jump[i,:]
#plotting simulation
timeline=np.arange(T)*dt
plt.plot(timeline,X_sim[:, :30])
plt.show()

#add seasonality

logP_sim_Q=theta_sim[:, None] + X_sim
P_sim_Q=np.exp(logP_sim_Q)
E_FUT_Q = np.mean(P_sim_Q, axis=1)
T2 = np.sum(~np.isnan(F_daily))
print("CORRECT T",T2)
#check
print(riskpremium_full)

#compare with market price future at exipration

fexp = np.zeros(future_df["EXPIRATION"].shape[0])

for i in range(len(future_df)):
    idx = date_sim == future_df["EXPIRATION"].iloc[i]

    if np.sum(idx) == 1:
        fexp[i] = E_FUT_Q[idx][0]

plt.figure(figsize=(12,6))

plt.plot(
    future_df["EXPIRATION"],
    future_df["F"].values,
    label="Market Futures"
)

plt.plot(
    future_df["EXPIRATION"],
    fexp,
    '*r',
    label="Risk-neutral expected futures"
)

plt.xlabel("Expiry")
plt.ylabel("Price")
plt.title("Market Futures vs Risk-neutral Simulation")
plt.legend()
plt.grid(True)

plt.show()



print(E_FUT[:10])
print(E_FUT_Q[:10])

print(F_daily_cl)
print(E_FUT_cl)

print(
    F_daily_cl / E_FUT_cl
)


df_check = pd.DataFrame({
    "Date": date_sim,
    "MeanPrice": E_FUT_Q
})

monthly = (
    df_check
    .set_index("Date")
    .resample("MS")
    .mean()
)

print(monthly)
print(future_df["EXPIRATION"])

print(mu)
print(k)
print(mu/k)
print(Xt[-1])

plt.plot(date_sim, theta_sim)
plt.show()

error=fexp-future_df["F"]
print(error)
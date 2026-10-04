"""Creates sample_car_data.csv (SYNTHETIC, same schema as CarDekho 'Car details v3.csv')
so the app can run offline. Replace with the real Kaggle file for your final submission."""
import numpy as np, pandas as pd
rng = np.random.default_rng(42); N = 5000
brands = {"Maruti":1.0,"Hyundai":1.15,"Honda":1.4,"Toyota":1.9,"Ford":1.2,"Mahindra":1.5,
          "Tata":1.0,"Volkswagen":1.3,"Renault":0.95,"BMW":4.5,"Mercedes-Benz":5.0,"Audi":4.2}
bn = rng.choice(list(brands), N, p=[.25,.18,.08,.08,.07,.08,.08,.05,.05,.03,.03,.02])
year = rng.integers(2005, 2021, N); age = 2026 - year
fuel = rng.choice(["Petrol","Diesel","CNG","LPG"], N, p=[.5,.45,.04,.01])
trans = rng.choice(["Manual","Automatic"], N, p=[.85,.15])
seller = rng.choice(["Individual","Dealer","Trustmark Dealer"], N, p=[.8,.15,.05])
owner = rng.choice(["First Owner","Second Owner","Third Owner","Fourth & Above Owner","Test Drive Car"], N, p=[.65,.27,.06,.015,.005])
km = np.clip(rng.normal(age*9000+10000, 15000), 1000, 400000).astype(int)
eng = np.clip(rng.normal(1250,350,N)*np.array([brands[b]**.3 for b in bn]),700,3000).astype(int)
pw = np.clip(eng*0.068+rng.normal(0,8,N),35,300).round(1)
mil = np.clip(26-eng/220+rng.normal(0,1.5,N),8,28).round(2)
seats = rng.choice([4,5,7,8], N, p=[.05,.8,.12,.03])
base = 220000*np.array([brands[b] for b in bn])*(pw/70)**1.1
price = base*np.exp(-0.11*age)*np.exp(-km/1.2e6)*np.where(trans=="Automatic",1.3,1)\
        *np.where(fuel=="Diesel",1.15,1)*np.where(owner=="First Owner",1,.88)*np.exp(rng.normal(0,.12,N))
df = pd.DataFrame({"name":[f"{b} Model{rng.integers(1,9)}" for b in bn],"year":year,
    "selling_price":(price//1000*1000).astype(int),"km_driven":km,"fuel":fuel,"seller_type":seller,
    "transmission":trans,"owner":owner,"mileage":[f"{m} kmpl" for m in mil],
    "engine":[f"{e} CC" for e in eng],"max_power":[f"{p} bhp" for p in pw],
    "torque":"190Nm@ 2000rpm","seats":seats.astype(float)})
df.loc[rng.choice(N,120,replace=False),["mileage","engine","max_power","seats"]] = np.nan
df = pd.concat([df, df.sample(60, random_state=1)]).reset_index(drop=True)
df.to_csv("sample_car_data.csv", index=False); print(df.shape)

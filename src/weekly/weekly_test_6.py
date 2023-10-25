sp500 = pd.read_parquet("C:/Users/MSI laptop/Downloads/sp500.parquet", engine = "fastparquet")

ff_factors = pd.read_parquet("C:/Users/MSI laptop/Downloads/ff_factors.parquet", engine = "fastparquet")

df_new = sp500.merge(ff_factors, on='Date', how='left')

df_new["Excess Return"] = df_new["Monthly Returns"]-df_new["RF"]

df_new = df_new.sort_values(by=['Date']).reset_index()

df_new["ex_ret_1"]=df_new.groupby("Symbol")["Excess Return"].shift(1)

df_new = df_new.dropna(subset = ['ex_ret_1'])

df_new = df_new.dropna(subset = ['HML'])

df_amzn = df_new[df_new["Symbol"]=="AMZN"]

df_amzn = df_amzn.drop("Symbol", axis = 1)

left_hand_side = df_amzn[["Excess Return"]]

right_hand_side = df_amzn[["Mkt-RF", "SMB", "HML"]]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stt


def fetch(symbol:str, market:str='USD') -> tuple[np.ndarray, np.ndarray]:
    url = f'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol={symbol}&market={market}&apikey=LN6AB4R2IAMHDV8I&datatype=csv'

    df = pd.read_csv(url)

    C = df['close'].to_numpy()

    L = np.log(C)

    return C, L

Ceth, Leth = fetch("ETH")
Cbtc, Lbtc = fetch("BTC")

Cs = np.polyfit(Lbtc, Leth, 1)
P = np.poly1d(Cs)

Relation = f'log(ETH) = {str(round(Cs[0], 2))}*log(BTC) + {str(round(Cs[1], 2))}'


plt.scatter(Lbtc, Leth, s=4)
plt.plot([10.9, 11.6], [P(10.9), P(11.6)], label="Linear Regression", linewidth=1.3, c='k')
plt.xlabel("log(BTC price)")
plt.ylabel("log(ETH price)")
plt.title(Relation)
plt.show()

PCC, _ = stt.pearsonr(Lbtc, Leth)
print("Pearson Correlation Result",PCC)

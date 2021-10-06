import backtrader as bt

if __name__ == '__main__':
    cerebro = bt.Cerebro()
    ibstore = bt.stores.IBStore()
    cerebro.broker = ibstore.getbroker()
    print('done')

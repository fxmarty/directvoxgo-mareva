_base_ = '../default.py'

expname = 'dvgo_nike'
basedir = './logs/mareva'

data = dict(
    datadir='./data/mareva/nike/',
    dataset_type='mareva',
    inverse_y=True,
    white_bkgd=True,
)

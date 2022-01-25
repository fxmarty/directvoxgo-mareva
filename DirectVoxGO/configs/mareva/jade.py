_base_ = '../default.py'

expname = 'dvgo_jade'
basedir = './logs/mareva'

data = dict(
    datadir='./data/mareva/jade/',
    dataset_type='mareva',
    inverse_y=True,
    white_bkgd=False,
)
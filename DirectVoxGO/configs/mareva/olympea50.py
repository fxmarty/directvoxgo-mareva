_base_ = '../default.py'

expname = 'dvgo_olympea50'
basedir = './logs/mareva'

data = dict(
    datadir='./data/mareva/olympea50/',
    dataset_type='mareva',
    inverse_y=True,
    white_bkgd=False,
)

_base_ = '../default.py'

expname = 'dvgo_cube50'
basedir = './logs/mareva'

data = dict(
    datadir='./data/mareva/cube50/',
    dataset_type='mareva',
    inverse_y=True,
    white_bkgd=True,
)

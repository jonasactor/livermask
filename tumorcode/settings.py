###
### process options and setup global variables
###

def process_options():

    from optparse import OptionParser # TODO update to ArgParser (python2 --> python3)

    parser = OptionParser()
    parser.add_option( "--hvd",
                  action="store_true", dest="with_hvd", default=False,
                  help="use horovod for multicore parallelism")
    parser.add_option( "--gpu",
                  type="int", dest="gpu", default=0,
                  help="number of gpus", metavar="int")
    parser.add_option( "--builddb",
                  action="store_true", dest="builddb", default=False,
                  help="load all training data into npy", metavar="FILE")
    parser.add_option( "--trainmodel",
                  action="store_true", dest="trainmodel", default=False,
                  help="train model on all data", metavar="bool")
    parser.add_option( "--predictmodel",
                  action="store", dest="predictmodel", default=None,
                  help="model weights (.h5) for prediction", metavar="Path")
    parser.add_option( "--predictimage",
                  action="store", dest="predictimage", default=None,
                  help="image to segment", metavar="Path")
    parser.add_option( "--segmentation",
                  action="store", dest="segmentation", default=None,
                  help="location for seg prediction output ", metavar="Path")
    parser.add_option( "--trainingsolver",
                  action="store", dest="trainingsolver", default='adam',
                  help="setup info", metavar="string")
    parser.add_option( "--dbfile",
                  action="store", dest="dbfile", default="./trainingdata.csv",
                  help="training data file", metavar="string")
    parser.add_option( "--trainingresample",
                  type="int", dest="trainingresample", default=256,
                  help="resample so that model prediction occurs at this resolution", metavar="int")
    parser.add_option( "--trainingbatch",
                  type="int", dest="trainingbatch", default=20,
                  help="batch size", metavar="int")
    parser.add_option( "--validationbatch",
                  type="int", dest="validationbatch", default=20,
                  help="batch size", metavar="int")
    parser.add_option( "--kfolds",
                  type="int", dest="kfolds", default=1,
                  help="perform kfold prediction with k folds", metavar="int")
    parser.add_option( "--idfold",
                  type="int", dest="idfold", default=-1,
                  help="individual fold for k folds", metavar="int")
    parser.add_option( "--rootlocation",
                  action="store", dest="rootlocation", default='/rsrch1/ip/jacctor/LiTS/LiTS',
                  help="root location for images for training", metavar="Path")
    parser.add_option("--numepochs",
                  type="int", dest="numepochs", default=10,
                  help="number of epochs for training", metavar="int")
    parser.add_option("--outdir",
                  action="store", dest="outdir", default='./',
                  help="directory for output", metavar="Path")
    parser.add_option( "--augment",
                  action="store_true", dest="augment", default=False,
                  help="use data augmentation for training", metavar="bool")
    parser.add_option( "--skip",
                  action="store_true", dest="skip", default=False,
                  help="skip connections in UNet", metavar="bool")
    parser.add_option( "--fanout",
                  action="store_true", dest="fanout", default=False,
                  help="fan out as UNet gets deeper (more filters at deeper levels)", metavar="bool")
    parser.add_option( "--batchnorm",
                  action="store_true", dest="batchnorm", default=False,
                  help="use batch normalization in UNet", metavar="bool")
    parser.add_option( "--reverse_up",
                  action="store_true", dest="reverse_up", default=False,
                  help="perform conv2D then upsample on way back up UNet", metavar="bool")
    parser.add_option( "--depth",
                  type="int", dest="depth", default=3,
                  help="number of down steps to UNet", metavar="int")
    parser.add_option( "--filters",
                  type="int", dest="filters", default=16,
                  help="number of filters for output of CNN layer", metavar="int")
    parser.add_option( "--activation",
                  action="store", dest="activation", default='relu',
                  help="activation function", metavar="string")
    parser.add_option( "--segthreshold",
                  type="float", dest="segthreshold", default=0.5,
                  help="cutoff for binary segmentation from real-valued output", metavar="float")
    parser.add_option( "--dropout",
                  type="float", dest="dropout", default=0.5,
                  help="percent  dropout", metavar="float")
    parser.add_option( "--nu",
                  type="int", dest="nu", default=2,
                  help="number of smoothing steps (conv blocks) at each down/up", metavar="int")
    parser.add_option( "--nu_bottom",
                  type="int", dest="nu_bottom", default=4,
                  help="number of conv blocks on bottom of UNet", metavar="int")
    parser.add_option( "--v",
                  type="int", dest="v", default=1,
                  help="number of v-cycles to complete", metavar="int")
    parser.add_option( "--regularizer",
                  type="float", dest="regularizer", default=0.0,
                  help="Lagrange multiplier on kernel regularization term in loss function", metavar="float")
    parser.add_option( "--makepredictions",
                  action="store_true", dest="makepredictions", default=False,
                  help="make predictions during k-fold training", metavar="bool")
    parser.add_option( "--rescon",
                  action="store_true", dest="rescon", default=False,
                  help="residual connection on blocks", metavar="bool")
    parser.add_option( "--rescon2",
                  action="store_true", dest="rescon2", default=False,
                  help="residual connections on blocks, with extra conv2d", metavar="bool")
    parser.add_option( "--hu_lb",
                  type="int", dest="hu_lb", default=-100,
                  help="lower bound for CT windowing", metavar="int")
    parser.add_option( "--hu_ub",
                  type="int", dest="hu_ub", default=300,
                  help="upper bound for CT windowing", metavar="int")
    parser.add_option( "--makedropoutmap",
                  action="store_true", dest="makedropoutmap", default=False,
                  help="perform multiple evaluations of predictions using different dropout draws", metavar="bool")
    parser.add_option( "--ntrials",
                  type="int", dest="ntrials", default=20,
                  help="number of Bernoulli trials for dropout draws", metavar="int")
    parser.add_option( "--lr",
                  type="float", dest="lr", default=0.001,
                  help="learning rate for Adam optimizer. Not used if not using Adam.", metavar="float")
    global options
    global args

    (options, args) = parser.parse_args()
    
    return (options, args)


def perform_setup(options):
    
    import numpy as np
    import sys
    import keras
    import keras.backend as K
    import tensorflow as tf

    sys.setrecursionlimit(5000)

    if options.with_hvd:
        import horovod.keras as hvd
        hvd.init()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        if options.gpu > 1:
            devlist = '0'
            for i in range(1,options.gpu):
                devlist += ','+str(i)
            config.gpu_options.visible_device_list = devlist
        else:
            config.gpu_options.visible_device_list = str(hvd.local_rank())
        K.set_session(tf.Session(config=config))


    global _globalnpfile
    global _globalexpectedpixel
    global IMG_DTYPE
    global SEG_DTYPE
    global FLOAT_DTYPE
    global _nx
    global _ny


    # raw dicom data is usually short int (2bytes) datatype
    # labels are usually uchar (1byte)
    IMG_DTYPE = np.int16
    SEG_DTYPE = np.uint8
    FLOAT_DTYPE = np.float32

    _globalnpfile = options.dbfile.replace('.csv','%d.npy' % options.trainingresample )
    _globalexpectedpixel=512
    _nx = options.trainingresample
    _ny = options.trainingresample

    return IMG_DTYPE, SEG_DTYPE, _globalnpfile, _globalexpectedpixel, _nx, _ny

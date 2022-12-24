import warnings
warnings.filterwarnings( 'ignore' )

from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import *
from tensorflow.keras import backend as K

from tensorflow.keras.layers import Input, Reshape, Embedding, BatchNormalization
from tensorflow.keras.layers import GRU, LSTM, Conv1D, Conv2D, Dense, Bidirectional
from tensorflow.keras.layers import SpatialDropout1D, Dropout, Concatenate, concatenate
from tensorflow.keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling1D, AveragePooling1D

def NB_Model(  ):
    model = BernoulliNB()
    return model

def LR_Model( pen='l2', c=1, sol='saga', class_weight='balanced' ):
    if pen == 'elasticnet':
        sol = 'saga'
    model = LogisticRegression( penalty=pen, C=c, class_weight=class_weight, solver=sol, n_jobs=-1 )
    return model

def RF_Model( n_est=500, crit='entropy', bootstrap=True, oob_score=True, class_weight='balanced' ):
    model = RandomForestClassifier( n_estimators=n_est, criterion=crit, n_jobs=-1,
                                   bootstrap=bootstrap, oob_score=oob_score, class_weight=class_weight )
    return model

def SVM_Model( c=1, ker='rbf', gam=0.5, max_iter=5000, class_weight='balanced' ):
    model = SVC( C=c, kernel=ker, gamma=gam, probability=True, class_weight=class_weight, max_iter=max_iter, verbose=False )
    return model

def CNN_George(
    max_len=100, embed_inp=10000, embed_size=300, embedding_matrix=[], embed_trainable=False,
    emb_weights_init='', optimizer='' ):
    
    filter_sizes = [ 3,4,5 ]
    num_filters = 32
    inp = Input( shape=( max_len, ) )
    if embedding_matrix == []:
        x = Embedding( input_dim=embed_inp, output_dim=embed_size,
                      embeddings_initializer=emb_weights_init, trainable=embed_trainable )( inp )
    else:
        x = Embedding( input_dim=embed_inp, output_dim=embed_size,
                      weights=[ embedding_matrix ], trainable=embed_trainable )( inp )
    
    x = Reshape( ( max_len, embed_size, 1 ) )( x )
    pooled = [  ]
    for j,i in enumerate( filter_sizes ):
        conv = Conv2D( num_filters, kernel_size=( i, embed_size ), activation='relu' )( x )
        globalmax = GlobalMaxPooling2D(  )( conv )
        pooled.append( globalmax )
    z = Concatenate( axis=1 )( pooled )
    output = Dense( 1, activation='sigmoid' )( z )
    
    model = Model( inputs=inp, outputs=output )
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    return model

def focal_loss( gamma=1., alpha=.1 ):
    def focal_loss_fixed( y_true, y_pred ):
        pt_1 = tf.where( tf.equal( y_true, 1 ), y_pred, tf.ones_like( y_pred ) )
        pt_0 = tf.where( tf.equal( y_true, 0 ), y_pred, tf.zeros_like( y_pred ) )
        return -K.sum( alpha * K.pow( 1. - pt_1, gamma ) * K.log( pt_1 ) ) - K.sum( ( 1 - alpha ) * K.pow( pt_0, gamma ) * K.log( 1. - pt_0 ) )
    return focal_loss_fixed

def BGRU_P(
    max_len=100, embed_inp=10000, embed_size=300, embedding_matrix=[], embed_trainable=False,
    spdrpt=.1, drpt=.1, emb_weights_init='', fc_weights_init='', fc_act='', optimizer='',
    fcl_loss_alp=.1, fcl_loss_gam=1 ):
    
    inp = Input( shape=( max_len, ) )
    if embedding_matrix == []:
        x = Embedding( input_dim=embed_inp, output_dim=embed_size,
                      embeddings_initializer=emb_weights_init, trainable=embed_trainable )( inp )
    else:
        x = Embedding( input_dim=embed_inp, output_dim=embed_size,
                      weights=[ embedding_matrix ], trainable=embed_trainable )( inp )
    
    x = SpatialDropout1D( spdrpt )( x )
    x1 = Bidirectional( GRU( 128, return_sequences=True ) )(x)
    x2 = Bidirectional( GRU( 64, return_sequences=True ) )(x)
    
    conc = concatenate( [ x1, x2 ] )
    
    pooled = [  ]
    
    avg_pool = GlobalAveragePooling1D(  )(conc)
    max_pool = GlobalMaxPooling1D(  )(conc)
    
    pooled.append( avg_pool )
    pooled.append( max_pool )
    
    x = Concatenate( axis=1 )( pooled )
    x = Dropout( drpt ) ( x )
    
    fc1 = Dense( 100, activation=fc_act, kernel_initializer=fc_weights_init )( x )
    fc2 = Dense( 50, activation=fc_act, kernel_initializer=fc_weights_init )( fc1 )
    output = Dense( 1, activation='sigmoid' )( fc2 )
    
    model = Model( inputs=inp, outputs=output )
    
    model.compile(
        loss=[ focal_loss( alpha=fcl_loss_alp, gamma=fcl_loss_gam ) ],
        optimizer=optimizer,
        metrics=['accuracy']
    )
    return model

def CNN_GRU(
    max_len=100, embed_inp=10000, embed_size=300, embedding_matrix=[], embed_trainable=False,
    drpt=.1, emb_weights_init='', optimizer='', ker_regularizer='' ):
    
    inp = Input( shape=( max_len, ) )
    if embedding_matrix == []:
        x = Embedding( input_dim=embed_inp, output_dim=embed_size,
                      embeddings_initializer=emb_weights_init, trainable=embed_trainable )( inp )
    else:
        x = Embedding( input_dim=embed_inp, output_dim=embed_size,
                      weights=[ embedding_matrix ], trainable=embed_trainable )( inp )
    
    x = Dropout( drpt )( x )
    conv = Conv1D( filters=100, kernel_size=4, padding='same', activation='relu',
                  kernel_regularizer=ker_regularizer )( x )
    pool = MaxPooling1D( pool_size=4 )( conv )
    gru = GRU( 100, return_sequences=True )( pool )
    z = GlobalMaxPooling1D(  )( gru )
    output = Dense( 1, activation='sigmoid', kernel_regularizer=ker_regularizer )( z )
    
    model = Model( inputs=inp, outputs=output )
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    return model

def BLSTM(
    max_len=100, embed_inp=10000, embed_size=300, embedding_matrix=[], embed_trainable=False,
    spdrpt=.1, drpt=.1, emb_weights_init='', fc_weights_init='', fc_act='', optimizer='', lstm_units=32 ):
    
    inp = Input( shape=( max_len, ) )
    if embedding_matrix == []:
        x = Embedding( input_dim=embed_inp, output_dim=embed_size,
                      embeddings_initializer=emb_weights_init, trainable=embed_trainable )( inp )
    else:
        x = Embedding( input_dim=embed_inp, output_dim=embed_size,
                      weights=[ embedding_matrix ], trainable=embed_trainable )( inp )
    
    x = SpatialDropout1D( spdrpt )( x )
    z = Bidirectional( LSTM( lstm_units ) )( x )
    z = Dropout( drpt )( z )
    fc1 = Dense( 100, activation=fc_act, kernel_initializer=fc_weights_init )( z )
    fc2 = Dense( 50, activation=fc_act, kernel_initializer=fc_weights_init )( fc1 )
    output = Dense( 1, activation='sigmoid' )( fc2 )
    
    model = Model( inputs=inp, outputs=output )
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    return model

def BGRU(
    max_len=100, embed_inp=10000, embed_size=300, embedding_matrix=[], embed_trainable=False,
    spdrpt=.1, drpt=.1, emb_weights_init='', fc_weights_init='', fc_act='', optimizer='', gru_units=32 ):
    
    inp = Input( shape=( max_len, ) )
    if embedding_matrix == []:
        x = Embedding( input_dim=embed_inp, output_dim=embed_size,
                      embeddings_initializer=emb_weights_init, trainable=embed_trainable )( inp )
    else:
        x = Embedding( input_dim=embed_inp, output_dim=embed_size,
                      weights=[ embedding_matrix ], trainable=embed_trainable )( inp )
    
    x = SpatialDropout1D( spdrpt )( x )
    z = Bidirectional( GRU( gru_units ) )( x )
    z = Dropout( drpt )( z )
    fc1 = Dense( 100, activation=fc_act, kernel_initializer=fc_weights_init )( z )
    fc2 = Dense( 50, activation=fc_act, kernel_initializer=fc_weights_init )( fc1 )
    output = Dense( 1, activation='sigmoid' )( fc2 )
    model = Model( inputs=inp, outputs=output )
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    return model

def CNN_RUT(
    max_len=100, embed_inp=10000, embed_size=300, embedding_matrix=[], embed_trainable=False,
    spdrpt=.1, drpt=.1, emb_weights_init='', conv_weights_init='', conv_act='', fc_weights_init='',
    fc_act='', optimizer='' ):
    
    filter_sizes = [ 1,2,3,4,5 ]
    num_filters = 32
    
    inp = Input( shape=( max_len, ) )
    if embedding_matrix == []:
        x = Embedding( input_dim=embed_inp, output_dim=embed_size,
                      embeddings_initializer=emb_weights_init, trainable=embed_trainable )( inp )
    else:
        x = Embedding( input_dim=embed_inp, output_dim=embed_size,
                      weights=[ embedding_matrix ], trainable=embed_trainable )( inp )
    
    x = SpatialDropout1D( spdrpt )( x )
    x = Reshape( ( max_len, embed_size, 1 ) )( x )
    pooled = [  ]
    for j,i in enumerate( filter_sizes ):
        conv = Conv2D( num_filters, kernel_size=( i, embed_size ), activation=conv_act )( x )
        globalmax = GlobalMaxPooling2D(  )( conv )
        pooled.append( globalmax )
    z = Concatenate( axis=1 )( pooled )
    z = Dropout( drpt )( z )
    fc1 = Dense( 100, activation=fc_act, kernel_initializer=fc_weights_init )( z )
    fc2 = Dense( 50, activation=fc_act, kernel_initializer=fc_weights_init )( fc1 )
    outp = Dense( 1, activation='sigmoid' )( fc2 )
    
    model = Model( inputs=inp, outputs=outp )
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    return model

def MCBiGRU(
    max_len=100, embed_inp=10000, embed_size=300, embedding_matrix=[], embed_trainable=False,
    drpt=.1, emb_weights_init='', conv_weights_init='', conv_act='', fc_weights_init='',
    fc_act='', optimizer='' ):
    
    window_sizes = [ 1,2,3,5,6 ]
    allchannels = [  ]
    
    inp = Input( shape=( max_len, ) )
    if embedding_matrix == []:
        x = Embedding( input_dim=embed_inp, output_dim=embed_size,
                      embeddings_initializer=emb_weights_init, trainable=embed_trainable )( inp )
    else:
        x = Embedding( input_dim=embed_inp, output_dim=embed_size,
                      weights=[ embedding_matrix ], trainable=embed_trainable )( inp )
    
    for j,i in enumerate( window_sizes ):
        conv = Conv1D( filters=128, kernel_size=i, padding='valid',
                      activation=conv_act, kernel_initializer=conv_weights_init )( x )
        drpt1 = Dropout( drpt )( conv )
        maxpool = MaxPooling1D( pool_size=4 )( drpt1 )
        bgru = Bidirectional( GRU( 200, return_sequences=False ) )( maxpool )
        drpt2 = Dropout( drpt )( bgru )
        allchannels.append( drpt2 )
    
    z = concatenate( allchannels )
    fc = Dense( 2000, activation=fc_act, kernel_initializer=fc_weights_init )( z )
    bnorm = BatchNormalization(  )( fc )
    output = Dense( 1, activation='sigmoid' )( bnorm )
    
    model = Model( inputs=inp, outputs=output )
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    return model

def CNN_gram(
    max_len=100, embed_inp=10000, embed_size=300, embedding_matrix=[], embed_trainable=False,
    drpt=.1, emb_weights_init='', conv_weights_init='', conv_act='', fc_weights_init='',
    fc_act='', optimizer='' ):
    
    maxpooloutput = [ ]
    avgpooloutput = [ ]
    
    inp = Input( shape=(max_len, ) )
    if embedding_matrix == []:
        x = Embedding( input_dim=embed_inp, output_dim=embed_size,
                      embeddings_initializer=emb_weights_init, trainable=embed_trainable )( inp )
    else:
        x = Embedding( input_dim=embed_inp, output_dim=embed_size,
                      weights=[ embedding_matrix ], trainable=embed_trainable )( inp )
    
    conv1 = Conv1D( filters=512, kernel_size=1, padding='same',
                   activation=conv_act, kernel_initializer=conv_weights_init )( x )
    maxpool1 = MaxPooling1D( pool_size=2 )( conv1 )
    avgpool1 = AveragePooling1D( pool_size=2 )( conv1 )
    maxpooloutput.append( maxpool1 )
    avgpooloutput.append( avgpool1 )
    
    conv2 = Conv1D( filters=512, kernel_size=2, padding='same',
                   activation=conv_act, kernel_initializer=conv_weights_init )( maxpool1 )
    maxpool2 = MaxPooling1D( pool_size=2 )( conv2 )
    avgpool2 = AveragePooling1D( pool_size=2 )( conv2 )
    maxpooloutput.append( maxpool2 )
    avgpooloutput.append( avgpool2 )
    
    conv3 = Conv1D( filters=512, kernel_size=1, padding='same',
                   activation=conv_act, kernel_initializer=conv_weights_init )( maxpool2 )
    maxpool3 = MaxPooling1D( pool_size=2 )( conv3 )
    avgpool3 = AveragePooling1D( pool_size=2 )( conv3 )
    maxpooloutput.append( maxpool3 )
    avgpooloutput.append( avgpool3 )
    
    conv4 = Conv1D( filters=512, kernel_size=1, padding='same',
                   activation=conv_act, kernel_initializer=conv_weights_init )( maxpool3 )
    maxpool4 = MaxPooling1D( pool_size=2 )( conv4 )
    avgpool4 = AveragePooling1D( pool_size=2 )( conv4 )
    maxpooloutput.append( maxpool4 )
    avgpooloutput.append( avgpool4 )
    
    allpoolings = []
    allpoolings.extend( maxpooloutput )
    allpoolings.extend( avgpooloutput )
    
    z = concatenate( allpoolings, axis=1 )
    
    gmaxpool = GlobalMaxPooling1D()( z )
    gavgpool = GlobalAveragePooling1D()( z )
    
    conc = concatenate( [ gmaxpool, gavgpool ] )
    
    fc1 = Dense( 512, activation=fc_act, kernel_initializer=fc_weights_init )( conc )
    drpt1 = Dropout( drpt )( fc1 )
    bnorm1 = BatchNormalization(  )( drpt1 )
    
    fc2 = Dense( 256, activation=fc_act, kernel_initializer=fc_weights_init )( bnorm1 )
    drpt2 = Dropout( drpt )( fc2 )
    bnorm2 = BatchNormalization(  )( drpt2 )
    
    output = Dense( 1, activation='sigmoid' )( bnorm2 )
    
    model = Model( inputs=inp, outputs=output )
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    return model

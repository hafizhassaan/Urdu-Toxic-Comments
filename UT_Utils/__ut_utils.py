import warnings
warnings.filterwarnings( 'ignore' )

import re
import numpy as np
from sklearn.metrics import f1_score
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K

def remove_diacritics( line ):
    line = re.sub( u"َ", "", line ) # remove fathah
    line = re.sub( u"ِ", "", line ) # remove kasrah
    line = re.sub( u"ُ", "", line ) # remove dammah
    line = re.sub( u"ً", "", line ) # remove tanwin al fathah
    line = re.sub( u"ٍ", "", line ) # remove tanwin al kasrah
    line = re.sub( u"ٌ", "", line ) # remove tanwin al dammah
    line = re.sub( u"ْ", "", line ) # remove sukun
    line = re.sub( u"ّ", "", line ) # remove shaddah
    line = re.sub( u"ٰ", "", line ) # remove kharha fathah
    line = re.sub( u"ٖ", "", line ) # remove kharha kasrah
    line = re.sub( u"ٗ", "", line ) # remove inverted dammah
    line = re.sub( u"ؔ", "", line ) # remove takhalus sign
    line = re.sub( "؎", "", line ) # remove verse sign
    line = re.sub( "؁", "", line ) # remove year sign
    line = re.sub( "ٓ", "", line ) # remove maddah
    line = re.sub( "ۤ", "", line ) # remove high maddah
    return line

def urdu_preprocessing( line ):
    line = line.lower()
    line = re.sub( r"http\S+", " ", line ) # remove urls
    line = re.sub( r"www.\S+", " ", line ) # remove urls
    
    line = re.sub( "آ", "آ", line ) # for alif mad normalization
    line = re.sub( "ك", "ک", line ) # for kaaf normalization
    line = re.sub( "ي", "ی", line ) # for yaa normalization
    
    line = re.sub( "[0-9]", " ", line ) # remove english digits
    line = re.sub( "۱|۲|۳|۴|۵|۶|۷|۸|۹|۰", " ", line ) # remove urdu digits
    
    line = remove_diacritics( line ) # remove all diacritics
    
    line = re.sub( r"[۔؛؟،:٪٭]", " ", line ) # remove urdu puncts
    line = re.sub( r"[.;?,:]", " ", line ) # remove english puncts
    
    line = re.sub( r"[!\"`‘'’#$%&*+-/<=>@^\\_|~\t\n({\[\]})]", " ", line ) # remove other puncts
    
    line = re.sub( "[a-z]", " ", line ) # remove english characters
    
    #line = re.sub( u"", "", line ) # template
    
    line = re.sub( ' +', ' ', line ) # remove extra spaces
    
    return line

def roman_preprocessing( line ):
    line = line.lower()
    line = re.sub( r"http\S+", " ", line ) # remove urls
    line = re.sub( r"www.\S+", " ", line ) # remove urls
    
    line = re.sub( "[0-9]", " ", line ) # remove english digits
    
    line = re.sub( r"[.;?,:]", " ", line ) # remove english puncts
    
    line = re.sub( r"[!\"`‘'’#$%&*+-/<=>@^\\_|~\t\n({\[\]})]", " ", line ) # remove other puncts
    
    #line = re.sub( u"", "", line ) # template
    
    line = re.sub( ' +', ' ', line ) # remove extra spaces
    
    return line

def optimize_threshold( y_true, y_pred ):
    thresholds = np.arange( 0.1, 0.9, 0.001 )
    vscores = np.zeros( thresholds.shape[ 0 ] )
    for th in range( thresholds.shape[ 0 ] ):
        vscores[ th ] = f1_score( y_true, ( y_pred >= thresholds[ th ] ).astype( 'int32' ) )
    return thresholds[ np.argmax( vscores ) ] # return threshold that has max val score

class F1_score_callback( Callback ):
    def __init__( self, val_data, filepath, patience=10, decay=True, decay_rate=0.1, decay_after=2 ):
        self.file_path = filepath
        self.patience = patience
        self.patience_counter = 0
        self.decay = decay
        self.decay_rate = decay_rate
        self.decay_after = decay_after
        self.validation_data = val_data
    
    def __optimize_threshold_for_f1( self, y_true, y_pred ):
        thresholds = np.arange( 0.1, 0.9, 0.001 )
        vscores = np.zeros( thresholds.shape[ 0 ] )
        for th in range( thresholds.shape[ 0 ] ):
            vscores[ th ] = f1_score( y_true, ( y_pred >= thresholds[ th ] ).astype( 'int32' ) )
        return thresholds[ np.argmax( vscores ) ]

    def on_train_begin( self, logs=None ):
        self.val_f1s = []
        self.best_val_f1 = np.NINF

    def on_epoch_end( self, epoch, logs={} ):
        val_predict = self.model.predict( self.validation_data[ 0 ] )
        val_targ = self.validation_data[ 1 ]
        
        threshold = self.__optimize_threshold_for_f1( val_targ, val_predict )
        
        _val_f1 = f1_score( val_targ, ( val_predict >= threshold ).astype( 'int32' ) )
        self.val_f1s.append(_val_f1)
        
        self.patience_counter = self.patience_counter + 1
        
        printstatement = 'Epoch: {:03d}'.format( epoch + 1 ) +\
        ' --LR: {:1.0e}'.format( K.get_value( self.model.optimizer.lr ) ) +\
        ' --MaxValF1: {:0.7f}'.format( max( self.val_f1s ) ) +\
        ' --CurValF1: {:0.7f}'.format( _val_f1 ) + ' --Patience: {:02d}'.format( self.patience_counter )
        
        if _val_f1 > self.best_val_f1:
            self.model.save( self.file_path, overwrite=True, include_optimizer=True, save_format='h5' )
            self.best_val_f1 = _val_f1
            printstatement = printstatement + ' --F1 improved: {:0.7f}'.format( self.best_val_f1 )
            self.patience_counter = 0
        
        print( printstatement )
        
        if ( (self.decay==True) & ((self.patience_counter % self.decay_after) == 0) & (self.patience_counter != 0) ):
            K.set_value( self.model.optimizer.lr, ( K.get_value( self.model.optimizer.lr ) * self.decay_rate ) )
        
        if self.patience_counter == self.patience:
            self.model.stop_training = True
            print( 'Training stopped due to the patience parameter.' + \
                  ' --Patience: {:02d}'.format( self.patience_counter ) )
        
        return

class F1_score_callback_HF( Callback ):
    def __init__( self, val_data, filepath, patience=10, decay=True, decay_rate=0.1, decay_after=2 ):
        self.file_path = filepath
        self.patience = patience
        self.patience_counter = 0
        self.decay = decay
        self.decay_rate = decay_rate
        self.decay_after = decay_after
        self.validation_data = val_data
    
    def __optimize_threshold_for_f1( self, y_true, y_pred ):
        thresholds = np.arange( 0.1, 0.9, 0.001 )
        vscores = np.zeros( thresholds.shape[ 0 ] )
        for th in range( thresholds.shape[ 0 ] ):
            vscores[ th ] = f1_score( y_true, ( y_pred >= thresholds[ th ] ).astype( 'int32' ) )
        return thresholds[ np.argmax( vscores ) ]

    def on_train_begin( self, logs=None ):
        self.val_f1s = []
        self.best_val_f1 = np.NINF

    def on_epoch_end( self, epoch, logs={} ):
        val_predict = self.model.predict( [ self.validation_data[ 0 ], self.validation_data[ 1 ] ] )
        val_targ = self.validation_data[ 2 ]
        
        threshold = self.__optimize_threshold_for_f1( val_targ, val_predict )
        
        _val_f1 = f1_score( val_targ, ( val_predict >= threshold ).astype( 'int32' ) )
        self.val_f1s.append(_val_f1)
        
        self.patience_counter = self.patience_counter + 1
        
        printstatement = 'Epoch: {:03d}'.format( epoch + 1 ) +\
        ' --LR: {:1.0e}'.format( K.get_value( self.model.optimizer.lr ) ) +\
        ' --MaxValF1: {:0.7f}'.format( max( self.val_f1s ) ) +\
        ' --CurValF1: {:0.7f}'.format( _val_f1 ) + ' --Patience: {:02d}'.format( self.patience_counter )
        
        if _val_f1 > self.best_val_f1:
            self.model.save( self.file_path, overwrite=True, include_optimizer=True, save_format='h5' )
            self.best_val_f1 = _val_f1
            printstatement = printstatement + ' --F1 improved: {:0.7f}'.format( self.best_val_f1 )
            self.patience_counter = 0
        
        print( printstatement )
        
        if ( (self.decay==True) & ((self.patience_counter % self.decay_after) == 0) & (self.patience_counter != 0) ):
            K.set_value( self.model.optimizer.lr, ( K.get_value( self.model.optimizer.lr ) * self.decay_rate ) )
        
        if self.patience_counter == self.patience:
            self.model.stop_training = True
            print( 'Training stopped due to the patience parameter.' + \
                  ' --Patience: {:02d}'.format( self.patience_counter ) )
        
        return

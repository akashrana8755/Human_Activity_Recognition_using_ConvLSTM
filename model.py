def create_convlstm_model():
    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), input_shape=(None, 64, 64, 1), padding='same', return_sequences=True))
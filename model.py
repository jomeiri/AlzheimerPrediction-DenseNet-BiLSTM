import tensorflow as tf
from tensorflow.keras.layers import Input, Conv3D, BatchNormalization, Activation, concatenate, GlobalAveragePooling3D, Dense, Bidirectional, LSTM, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

def conv_block(x, filters, kernel_size, name):
    """Convolutional block with Batch Normalization and PReLU activation."""
    x = BatchNormalization(name=f'{name}_bn')(x)
    x = Activation('relu', name=f'{name}_relu')(x) # Thesis mentions PReLU but Keras 'relu' is common. Adjust if using custom PReLU.
    x = Conv3D(filters, kernel_size, padding='same', use_bias=False, name=f'{name}_conv')(x)
    return x

def dense_block(x, num_layers, growth_rate, name):
    """Dense Block as described in DenseNet."""
    for i in range(num_layers):
        # Bottleneck layer (Conv1x1x1 then Conv3x3x3)
        y = conv_block(x, 4 * growth_rate, (1, 1, 1), name=f'{name}_layer{i}_bottleneck') # 4*growth_rate filters for bottleneck
        y = conv_block(y, growth_rate, (3, 3, 3), name=f'{name}_layer{i}_conv')
        x = concatenate([x, y], axis=-1, name=f'{name}_layer{i}_concat')
    return x

def transition_layer(x, compression_factor, name, dropout_rate=0.0):
    """Transition layer for DenseNet to reduce feature map size."""
    filters = int(x.shape[-1] * compression_factor) # Reduce filters by compression factor
    x = BatchNormalization(name=f'{name}_bn')(x)
    x = Activation('relu', name=f'{name}_relu')(x)
    x = Conv3D(filters, (1, 1, 1), padding='same', use_bias=False, name=f'{name}_conv1x1')(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate, name=f'{name}_dropout')(x) # Thesis mentions dropout in transition layer [cite: 636]
    x = Conv3D(filters, (3, 3, 3), strides=(2, 2, 2), padding='same', use_bias=False, name=f'{name}_conv3x3_stride2')(x)
    return x

def build_densenet_3d(input_shape, num_dense_blocks, growth_rate, compression_factor=0.5):
    """
    Builds a 3D DenseNet architecture (simplified based on the thesis description).
    The exact DenseNet-121 structure (Table 2.1) needs precise layer counts.
    """
    img_input = Input(shape=input_shape, name='mri_input')

    # Initial Convolution
    x = Conv3D(64, (7, 7, 7), strides=(2, 2, 2), padding='same', use_bias=False, name='initial_conv')(img_input)
    x = BatchNormalization(name='initial_bn')(x)
    x = Activation('relu', name='initial_relu')(x)
    # x = MaxPooling3D((3, 3, 3), strides=(2, 2, 2), padding='same')(x) # Thesis structure doesn't explicitly show this after initial conv

    # Dense Blocks and Transition Layers
    num_layers_per_block = [2, 2, 2, 2] # Simplified, adjust based on Table 2.1 for DenseNet-121 (e.g., [6, 12, 24, 16])
    filters = 64 # Initial filters after first conv

    for i in range(num_dense_blocks):
        x = dense_block(x, num_layers_per_block[i], growth_rate, name=f'dense_block_{i+1}')
        filters += num_layers_per_block[i] * growth_rate # Update filters for next transition
        if i < num_dense_blocks - 1: # No transition after the last dense block [cite: 637]
            x = transition_layer(x, compression_factor, name=f'transition_layer_{i+1}')

    # Final Batch Norm + Activation
    x = BatchNormalization(name='final_bn')(x)
    x = Activation('relu', name='final_relu')(x)

    # Global Average Pooling (to get a fixed-size feature vector regardless of input spatial dims)
    x = GlobalAveragePooling3D(name='densenet_output')(x)

    densenet_model = Model(img_input, x, name='DenseNet3D_Feature_Extractor')
    return densenet_model

def build_densenet_bilstm_model(densenet_config, bilstm_units, num_classes, time_points, dropout_rate=0.0, l2_reg_lambda=0.0):
    """
    Builds the combined Hybrid DenseNet-BiLSTM model as proposed in the thesis.
    [cite: 820, 848]
    """
    # Input for longitudinal MRI images (e.g., 7 time points, each a 3D MRI)
    # The input shape here is (num_time_points, H, W, D, Channels)
    # Keras functional API expects a specific shape for the Input layer.
    # We will pass a single (H,W,D,C) image to DenseNet and then concatenate outputs.
    # The time dimension will be handled by a TimeDistributed wrapper or a custom loop in model training.
    # For a Keras functional model, we need to handle the sequence.
    
    # Input for each time point image
    input_per_time_point = Input(shape=densenet_config['input_shape'], name='input_image_per_time_point')

    # DenseNet feature extractor (same instance for all time points)
    densenet_feature_extractor = build_densenet_3d(
        input_shape=densenet_config['input_shape'],
        num_dense_blocks=densenet_config['num_dense_blocks'],
        growth_rate=densenet_config['growth_rate']
    )

    # Use TimeDistributed to apply DenseNet to each time point image
    # The overall input to the model will be (None, TIME_POINTS, H, W, D, C)
    longitudinal_input = Input(shape=(time_points,) + densenet_config['input_shape'], name='longitudinal_mri_input')
    
    # Apply DenseNet to each time slice
    # This will result in an output of shape (None, TIME_POINTS, DenseNet_Feature_Dim)
    densenet_features_sequence = tf.keras.layers.TimeDistributed(densenet_feature_extractor, name='time_distributed_densenet')(longitudinal_input) [cite: 871]

    # BiLSTM layer to capture temporal dependencies
    # Input to BiLSTM will be (batch_size, time_points, features_dim)
    bilstm_output = Bidirectional(LSTM(bilstm_units, return_sequences=False), name='bilstm_layer')(densenet_features_sequence) [cite: 871, 675]

    # Flatten the BiLSTM output (if return_sequences=True, then flatten each sequence)
    # Since return_sequences=False, BiLSTM directly outputs a 2D tensor for the last timestep.
    # If return_sequences=True, then Flatten(TimeDistributed(Flatten(...))) would be needed.
    # However, the thesis diagram shows concatenation and then Fully Connected, implying single output from BiLSTM.

    # Classification Layer [cite: 888]
    # Apply L2 regularization to the dense layer [cite: 986, 1153]
    output = Dense(num_classes, activation='softmax', name='classification_output',
                   kernel_regularizer=regularizers.l2(l2_reg_lambda))(bilstm_output) [cite: 892]

    # Combine into a single model
    model = Model(inputs=longitudinal_input, outputs=output, name='Hybrid_DenseNet_BiLSTM_AD_Predictor')

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), [cite: 964, 1148]
                  loss='sparse_categorical_crossentropy', # For integer labels like 0, 1, 2 [cite: 894]
                  metrics=['accuracy']) [cite: 1168]
    return model

# Example usage (for testing model construction)
if __name__ == '__main__':
    # Dummy input shapes for testing model compilation
    dummy_input_shape_3d = (64, 64, 64, 1) # Example, adjust as per your data
    dummy_time_points = 7
    dummy_num_classes = 2 # AD/CN

    dummy_densenet_config = {
        'input_shape': dummy_input_shape_3d,
        'num_dense_blocks': 4,
        'growth_rate': 16
    }

    dummy_model = build_densenet_bilstm_model(
        densenet_config=dummy_densenet_config,
        bilstm_units=256,
        num_classes=dummy_num_classes,
        time_points=dummy_time_points,
        dropout_rate=0.1,
        l2_reg_lambda=0.01
    )
    dummy_model.summary()

    # You can also generate dummy data to test a forward pass
    dummy_longitudinal_data = np.random.rand(1, dummy_time_points, *dummy_input_shape_3d)
    dummy_predictions = dummy_model.predict(dummy_longitudinal_data)
    print(f"Dummy prediction shape: {dummy_predictions.shape}") # Should be (1, num_classes)

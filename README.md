# Conintues Recurrent Neural Network
- Time series libraries contain RNN, CTRNN, and MEMS RNN.
- We used Python 3.8 and TensorFlow 2.8.
- We changed the Recurrent.py file to support CTRNN, MEMS
- The following classes were added to Recurrent.py: SimpleMEMSCTRNNCell, SimpleMEMSCTRNN, SimpleCTRNNCell, and SimpleCTRNN     to support the MEMS and CTRNN network.
- We used the TensorFlow auto differential for the backpropagation. 
- SimpleMEMSCTRNNCell parameters represents the MEMS manufactured, you could refers to MEMS CAS CTRNN NOT NORMALIZED APPROXIMATION.pdf for more details.
- The call function is the core function for each cell, so it calls every time the forward pass is called, and the TensorFlow auto differentiates this function.

- To use the new cells you only need to define your model using the new class,here is an example
```sh
    from keras import activations, Sequential, regularizers
    import tensorflow as tf
    from MEMSCTRNNLIB.Recurrent import SimpleMEMSCTRNN
    input_layer = Input((None, 1))
    alpha_=1 * (10 ** 5)
    displacement_=5.0 * (10 ** -6)
    stopper_=.5 * (10 ** -6)
    z_factor_=0.89921
    v_factor_=1.9146 * (10 ** -20)
    layer1 = SimpleMEMSCTRNN(5, activation=None, return_sequences=True,alpha=alpha_,
               displacement=displacement_,
               stopper=stopper_,
               z_factor=z_factor_,
               v_factor=v_factor_)(input_layer)

    layer2 = SimpleMEMSCTRNN(2, activation=None, return_sequences=True,alpha=alpha_,
               displacement=displacement_,
               stopper=stopper_,
               z_factor=z_factor_,
               v_factor=v_factor_)(layer1)
    output_ = SimpleMEMSCTRNN(1, activation=None, return_state=True,alpha=alpha_,
               displacement=displacement_,
               stopper=stopper_,
               z_factor=z_factor_,
               v_factor=v_factor_)(layer2)
    model = tf.keras.Model(inputs=input_layer, outputs=output_)
  
```

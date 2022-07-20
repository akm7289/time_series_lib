# Conintues Recurrent Neural Network
- Time series libraries contain RNN, CTRNN, and MEMS RNN.
- We used Python 3.8 and TensorFlow 2.8.
- We changed the Recurrent.py file to support CTRNN, MEMS
- The following classes were added to Recurrent.py: SimpleMEMSCTRNNCell, SimpleMEMSCTRNN, SimpleCTRNNCell, and SimpleCTRNN     to support the MEMS and CTRNN network.
- We used the TensorFlow auto differential for the backpropagation. 
- SimpleMEMSCTRNNCell parameters represents the MEMS manufactured, you could refers to MEMS CAS CTRNN NOT NORMALIZED APPROXIMATION.pdf for more details.
- The call function is the core function for each cell, so it calls every time the forward pass is called, and the TensorFlow auto differentiates this function.

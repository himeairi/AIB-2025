Technique Used
    - Vision Transformer
    - Schedule Sampling (Decrease teacher forcing ratio over epochs)
    - Mixed Precision (Accelerate cuda computing)
    - Attention Mechanism
    - Start Of Sequence (SOS) Cheat
    
Preprocessed Dataset
    - plt.grid(False)
    - plt.axis('off')
    - plt.figure(figsize=(672/100, 224/100))
    - Normalise data in csv to (0,1) scale for easy prediction

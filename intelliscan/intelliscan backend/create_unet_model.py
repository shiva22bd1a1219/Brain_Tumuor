import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.models import Model
from math import log2

cf = {
    "image_size": 256,
    "num_channels": 3,
    "num_layers": 12,
    "hidden_dim": 128,
    "mlp_dim": 32,
    "num_heads": 6,
    "dropout_rate": 0.1,
    "patch_size": 16,
}

cf["num_patches"] = (cf["image_size"]**2)//(cf["patch_size"]**2)

def mlp(x):
    x = L.Dense(cf["mlp_dim"], activation="gelu")(x)
    x = L.Dropout(cf["dropout_rate"])(x)
    x = L.Dense(cf["hidden_dim"])(x)
    return x

def transformer_encoder(x):
    skip = x
    x = L.LayerNormalization()(x)
    x = L.MultiHeadAttention(
        num_heads=cf["num_heads"],
        key_dim=cf["hidden_dim"]
    )(x, x)
    x = L.Add()([x, skip])

    skip = x
    x = L.LayerNormalization()(x)
    x = mlp(x)
    x = L.Add()([x, skip])
    return x

def build_model():
    inp = L.Input((cf["num_patches"], 768))
    x = L.Dense(cf["hidden_dim"])(inp)

    for _ in range(cf["num_layers"]):
        x = transformer_encoder(x)

    x = L.Dense(1, activation="sigmoid")(x)
    return Model(inp, x)

model = build_model()

model.save("best_model.keras")

print("UNet model created successfully")
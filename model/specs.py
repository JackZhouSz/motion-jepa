"""Named MotionJEPA encoder size specifications."""

MODEL_SPECS = {
    "tiny":   {"embed_dim": 192,  "depth": 6,  "num_heads": 3},
    "small":  {"embed_dim": 256,  "depth": 8,  "num_heads": 4},
    "base":   {"embed_dim": 384,  "depth": 12, "num_heads": 6},
    "large":  {"embed_dim": 512,  "depth": 16, "num_heads": 8},
    "huge":   {"embed_dim": 768,  "depth": 24, "num_heads": 12},
    "giant":  {"embed_dim": 1024, "depth": 32, "num_heads": 16},
}

PREDICTOR_SPECS = {
    "tiny":   {"predictor_embed_dim": 128, "depth": 4,  "num_heads": 4},
    "small":  {"predictor_embed_dim": 128, "depth": 4,  "num_heads": 4},
    "base":   {"predictor_embed_dim": 192, "depth": 6,  "num_heads": 6},
    "large":  {"predictor_embed_dim": 256, "depth": 8,  "num_heads": 8},
    "huge":   {"predictor_embed_dim": 384, "depth": 10, "num_heads": 12},
    "giant":  {"predictor_embed_dim": 384, "depth": 12, "num_heads": 12},
}

__all__ = ["MODEL_SPECS", "PREDICTOR_SPECS"]

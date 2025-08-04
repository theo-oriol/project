
def classifier(m):

    if m == "dinov2_vitl14_reg":
        from model.dino_model import dinov2_vitl14_reg as Classifier
        return Classifier()
    elif m == "dinov2_vitl14":
        from model.dino_model import dinov2_vitl14 as Classifier
        return Classifier()
    else : raise ValueError(f"Unknown model type: {m}")
    
 
def Show_model_layer_names(model):
    for layer in model.layers:
        print(layer.name)


def Show_model_weights(model):
    for layer in model.layers:
        print(layer.name)
        for weight in layer.weights:
            print("   ", weight.shape)

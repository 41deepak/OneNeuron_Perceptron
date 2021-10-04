from utils.model import Perceptron
from utils.all_utils import prepare_data, save_model, save_plot
import pandas as pd
import numpy as np

def main(data, eta, epochs, filename, plotname):
    df_OR = pd.DataFrame(data)
    print(df_OR)

    X,y = prepare_data(df_OR)
    
    model_OR = Perceptron(eta=eta,  epochs=epochs)
    model_OR.fit(X,y)
    _ = model_OR.total_loss()   #Dummy variable

    save_model(model_OR, filename=filename)
    save_plot(df_OR, plotname, model_OR)


if __name__ == "__main__":     #entry point
    OR = {
    "x1":[0,0,1,1],
    "x2":[0,1,0,1],
    "y":[0,1,1,1],}

    ETA = 0.3       # 0 to 1
    EPOCHS = 10

    main(data=OR, eta=ETA, epochs=EPOCHS, filename="OR.model", plotname="OR.png")
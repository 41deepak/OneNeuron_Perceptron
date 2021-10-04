from utils.model import Perceptron
from utils.all_utils import prepare_data, save_model, save_plot
import pandas as pd
import numpy as np

def main(data, eta, epochs, filename, plotname):
    df_and = pd.DataFrame(data)
    print(df_and)

    X,y = prepare_data(df_and)
    
    model_and = Perceptron(eta=eta,  epochs=epochs)
    model_and.fit(X,y)
    _ = model_and.total_loss()   #Dummy variable

    save_model(model_and, filename=filename)
    save_plot(df_and, plotname, model_and)


if __name__ == "__main__":     #entry point
    AND = {
    "x1":[0,0,1,1],
    "x2":[0,1,0,1],
    "y":[0,0,0,1],}

    ETA = 0.3       # 0 to 1
    EPOCHS = 10

    main(data=AND, eta=ETA, epochs=EPOCHS, filename="And.model", plotname="And.png")
from utils.model import Perceptron
from utils.all_utils import pd
from utils.all_utils import prepare_data


AND = {
    "x1":[0,0,1,1],
    "x2":[0,1,0,1],
    "y":[0,0,0,1],}

df_and = pd.DataFrame(AND)
df_and

X,y = prepare_data(df_and)
ETA = 0.3       # 0 to 1
EPOCHS = 10
model_and = Perceptron( eta = ETA, epochs= EPOCHS)
model_and.fit(X,y)

_ = model_and.total_loss()

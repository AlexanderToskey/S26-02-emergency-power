
# Imports
from occurrence_model import OutageOccurenceModel

# TODO
X_train = NotImplemented
y_train = NotImplemented
X_new = NotImplemented

model = OutageOccurenceModel()
model.train(X_train, y_train)
model.save("outage_occurrence_model.pkl")

predictions, probabilities = model.predict(X_new)


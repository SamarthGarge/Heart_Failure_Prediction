import pickle

knn = ...

pickle_out = open("model.pkl","wb")
pickle.dump(knn, pickle_out)

pickle_out.close()


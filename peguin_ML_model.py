import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from  sklearn.model_selection import train_test_split
import pickle

penguin_df = pd.read_csv("penguins.csv")

penguin_df.dropna(inplace=True)

#output=target/prediction, features=parameter buat modelling/prediction,
# output= result/target yabf diramalkan
output=penguin_df['species']
features = penguin_df[['island', 'bill_length_mm',  'bill_depth_mm', 'flipper_length_mm',
'body_mass_g', 'sex']]

#original after cleaning NaN values
#print(output.tail()) # dislplay 5 rows
#print(features.tail())

# features after encoding-method encoding
features = pd.get_dummies(features)
#print()
#print(features.tail())

output, uniques = pd.factorize(output)
#print(uniques)

#train test train_test_split
x_train, x_test, y_train, y_test = train_test_split(features, output,test_size=0.2)
rfc = RandomForestClassifier(random_state=123) #random state = parameter rfc; 80% data akan dopilih secara random
rfc.fit(x_train,y_train)

#buat prediction
y_pred= rfc.predict(x_test) # predict pada test data
score = accuracy_score(y_pred, y_test)

print('ur accuracy score for this model is {}'.format(score))

print('success')

#save the penguin RF RandomForestClassifier
rfc_pickle = open("random_forest_penguin.pickle",'wb') # create fail
pickle.dump(rfc, rfc_pickle) # rfc_pickle adalah bag, rfc adalah baju
rfc_pickle.close()


output_pickle = open('output_pickle.pickle', 'wb')
pickle.dump(uniques, output_pickle)
output_pickle.close()

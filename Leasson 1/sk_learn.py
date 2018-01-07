
from sklearn import tree

from sklearn import neighbors

from sklearn import discriminant_analysis

from sklearn import ensemble



#x=[height,width,shoesize]
x = [[181,80,44],[177,70,43],[160,60,38],
     [154,54,37] ,[166,65,40],[190,90,47],
     [175,64,39],[177,70,40],[159,55,37],
     [171,75,42],[181,85,43]]

y = ['male', 'male', 'female', 
     'female', 'male', 'male', 
     'female', 'female','female',
     'male', 'male']



DecisionTreeClassifier = tree.DecisionTreeClassifier()
DecisionTreeClassifier = DecisionTreeClassifier.fit(x,y)
print('DecisionTreeClassifier : ',DecisionTreeClassifier.predict([[190, 70, 43]]) ,'Score ' ,DecisionTreeClassifier.score(x,y))


ExtraTreeClassifier=tree.ExtraTreeClassifier()
ExtraTreeClassifier = ExtraTreeClassifier.fit(x,y)
print('ExtraTreeClassifier : ',ExtraTreeClassifier.predict([[190, 70, 43]]) ,'Score ' ,ExtraTreeClassifier.score(x,y))


KNeighborsClassifier=neighbors.KNeighborsClassifier();
KNeighborsClassifier = KNeighborsClassifier.fit(x,y)
print('KNeighborsClassifier : ',KNeighborsClassifier.predict([[190, 70, 43]]),'Score ' ,KNeighborsClassifier.score(x,y))

LinearDiscriminantAnalysis=discriminant_analysis.LinearDiscriminantAnalysis()
LinearDiscriminantAnalysis = LinearDiscriminantAnalysis.fit(x,y)
print('LinearDiscriminantAnalysis : ',LinearDiscriminantAnalysis.predict([[190, 70, 43]]),'Score ' ,LinearDiscriminantAnalysis.score(x,y))

ExtraTreesClassifier=ensemble.ExtraTreesClassifier()
ExtraTreesClassifier = ExtraTreesClassifier.fit(x,y)
print('ExtraTreesClassifier : ' ,ExtraTreesClassifier.predict([[190, 70, 43]]),'Score ' ,ExtraTreesClassifier.score(x,y))

RandomForestClassifier=ensemble.RandomForestClassifier()
RandomForestClassifier = RandomForestClassifier.fit(x,y)
print('RandomForestClassifier : ' ,RandomForestClassifier.predict([[190, 70, 43]]),'Score ' ,RandomForestClassifier.score(x,y))














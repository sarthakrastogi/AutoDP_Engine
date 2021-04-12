import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets

def app():
    numerical_cols = pd.read_csv('numerical_cols.csv')
    categorical_cols = pd.read_csv('categorical_cols.csv')

    if os.path.exists('numerical_cols.csv'):
        os.remove('numerical_cols.csv')
    if os.path.exists('categorical_cols.csv'):
        os.remove('categorical_cols.csv')

    #picking an algorithm


    #@st.cache(suppress_st_warning=True)
    def pick_algorithm():
        r_s = 2

        if target_feature in categorical_cols_list:
            task = 'classification'
            st.write("Pick a classification algorithm. Or pick multiple, and we'll tell you which one's working best.")
            clf_model_names = st.multiselect('Pick classification algorithms to apply:',['Logistic Regression', 'Random Forest Classifier', 'K Neighbours Classifier', 'Gradient Boosting Classifier', 'Support Vector Classifier', 'Extra Trees Classifier', 'AdaBoost Classifier', 'Gaussian Naive Bayes Classiifer', 'Gaussian Process Classifier', 'Bagging Classifier'])

            models = []

            if 'Logistic Regression' in clf_model_names:
                from sklearn.linear_model import LogisticRegression
                lr_clf = LogisticRegression()
                models.append(lr_clf)

            if 'Random Forest Classifier' in clf_model_names:
                from sklearn.ensemble import RandomForestClassifier
                rnd_clf = RandomForestClassifier(random_state=1)
                models.append(rnd_clf)


            from sklearn.neighbors import KNeighborsClassifier


            from sklearn.linear_model import PassiveAggressiveClassifier
            from sklearn.linear_model import RidgeClassifierCV
            from sklearn.linear_model import SGDClassifier
            from sklearn.linear_model import Perceptron

            from sklearn.svm import SVC
            from sklearn.svm import NuSVC
            from sklearn.svm import LinearSVC

            from sklearn.naive_bayes import GaussianNB
            from sklearn.naive_bayes import BernoulliNB

            from sklearn.gaussian_process import GaussianProcessClassifier

            from sklearn.tree import DecisionTreeClassifier


            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.ensemble import ExtraTreesClassifier
            from sklearn.ensemble import AdaBoostClassifier

            from sklearn.ensemble import BaggingClassifier
            from sklearn.ensemble import VotingClassifier




            knn = KNeighborsClassifier()
            gbc = GradientBoostingClassifier()
            svc = SVC(probability=True)
            ext = ExtraTreesClassifier()
            ada = AdaBoostClassifier()
            gnb = GaussianNB()
            gpc = GaussianProcessClassifier()
            bag = BaggingClassifier()
            pac = PassiveAgressiveClassifier()
            rdc = RidgeClassifierCV()
            sgd = SGDClassifier()
            prc = Preceptron()



            #models = [ran, knn, log, gbc, svc, ext, ada, gnb, gpc, bag, pac]
            models = []
            scores = pd.DataFrame(columns = ['Model', 'Score'])






            from sklearn.model_selection import cross_val_score
            from sklearn.model_selection import GridSearchCV
            from sklearn.model_selection import cross_val_predict
            from sklearn import model_selection

            for classifier in models:
                classifier.fit(X_train, y_train)
                acc = cross_val_score(classifier, X_train, y_train, scoring = "accuracy", cv = 2)
                scores.append({'Model' : classifier, 'Score' : acc.mean()}, ignore_index = True)



                #----------------------------------------------------------------------------------------------------------
                #Regression Models


        elif target_feature in numerical_cols_list:
            task = 'regression'
            st.write("Pick a regression algorithm. Or pick multiple, and we'll tell you which one's working best.")
            reg_models_names = st.multiselect('Pick regression algorithms to apply:',['ElasticNet', 'ElasticNetCV', 'CatBoost', 'Gradient Boost', 'Random Forest',
            'AdaBoost', 'Extra-Trees', 'Support Vector Machines', 'Ridge Regression', 'Ridge CV', 'Bayesian Ridge', 'Decision Tree', 'K-Nearest Neighbours', 'Lasso Regression',
            'Kernel Ridge', 'CCA', 'Multilayer Perceptron', 'Huber Regressor', 'RANSAC', 'Passive Agressive Regressor'])

            models = []

            if 'AdaBoost' in reg_models_names:
                from sklearn.ensemble import AdaBoostRegressor
                ada_reg = AdaBoostRegressor(random_state=r_s)
                models.append(ada_reg)

            if 'ElasticNet' in reg_models_names:
                from sklearn.linear_model import ElasticNet
                els_reg = ElasticNet(alpha=0.001,l1_ratio=0.70,max_iter=100,tol=0.01, random_state=r_s)
                models.append(els_reg)





    #TO PROCESS:
            from sklearn.metrics import mean_squared_error,mean_absolute_error
            from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor,ExtraTreesRegressor
            #HistGradientBoostingRegressor
            #from lightgbm import LGBMRegressor
            from catboost import CatBoostRegressor
            #from xgboost import XGBRegressor
            from sklearn.linear_model import Ridge,RidgeCV,BayesianRidge,LinearRegression,Lasso,LassoCV,RANSACRegressor,HuberRegressor,PassiveAggressiveRegressor,ElasticNetCV
            from sklearn.neighbors import KNeighborsRegressor
            from sklearn.tree import DecisionTreeRegressor
            from sklearn.ensemble import VotingRegressor
            from sklearn.svm import SVR
            from sklearn.kernel_ridge import KernelRidge
            from sklearn.cross_decomposition import CCA
            from sklearn.neural_network import MLPRegressor

            r_s = 2


            ecv_reg = ElasticNetCV(l1_ratio=0.9,max_iter=100,tol=0.01,random_state=r_s)
            cat_reg = CatBoostRegressor(logging_level='Silent',random_state=r_s)
            gb_reg = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber',random_state =r_s)
            #lgb_reg = LGBMRegressor(objective='regression', num_leaves=4, learning_rate=0.01,n_estimators=5000,max_bin=200,bagging_fraction=0.75, bagging_freq=5,  bagging_seed=7,    feature_fraction=0.2, feature_fraction_seed=7,verbose=-1, random_state=r_s )
            rf_reg = RandomForestRegressor(random_state=r_s)

            ext_reg = ExtraTreesRegressor(random_state=r_s)
            sv_reg = SVR(C= 20, epsilon= 0.008, gamma=0.0003)
            rid_reg = Ridge(alpha=6)
            rdcv_reg = RidgeCV()
            brd_reg = BayesianRidge()
            dt_reg = DecisionTreeRegressor()
            #lr_reg = LinearRegression()
            kn_reg = KNeighborsRegressor()
            las_reg = Lasso(alpha=0.00047,random_state=r_s)
            lscv_reg = LassoCV()
            krd_reg = KernelRidge()
            cca_reg = CCA()
            mlp_reg = MLPRegressor(random_state=r_s)
            #hg_reg = HistGradientBoostingRegressor(random_state=r_s)
            hub_reg = HuberRegressor()
            rns_reg = RANSACRegressor(random_state=r_s)
            psag_reg = PassiveAggressiveRegressor(random_state=r_s)
            #xgb = XGBRegressor(random_state=r_s)


            scores = pd.DataFrame(columns = ['Model', 'Score']) #ADD MORE SCORE METRICS
            count = 0
            for regressor in models: #CHANGE THE FOR LOOP THEY'RE ALL TRAINING AGAIN WHEN YOU TRAIN A NEW MODEL | MAYBE ADD A BUTTON TO CLICK TO CLICK BEFORE TRAINING
                st.write('Training', regressor)
                regressor.fit(X_train, y_train)
                acc = regressor.score(X_test, y_test)
                scores = scores.append({'Model' : regressor, 'Score' : acc}, ignore_index = True)
                #scores = scores.loc[len(scores.index)] = [regressor, acc]

                scores = scores.sort_values(by = 'Score', ascending = False)


            st.write(scores)

        #st.write()RECOMMEND THE MODEL(S) WITH THE HIGHEST ACCURACY


        st.write('Pick the model you want to work with.')

        model = st.selectbox("You can only pick one model. In later versions you'll be able to ensemble multiple models.",
                            models)


    pick_algorithm()



    numerical_cols.to_csv('numerical_cols.csv')
    categorical_cols.to_csv('categorical_cols.csv')



    st.write('---')

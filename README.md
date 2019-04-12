# stepwise-regression
stepwise regression can be used in feature selection. features will be ranked according to its importance

class stepwise_feature_selection():

    def find_best_feature(self,data,feature_opt,residual): #找出与residual相关性最高的feature
        max_co = 0
        best_feature = feature_opt[0]
        for i in feature_opt:
            corref = data[i].corr(residual['residual'])
            if abs(corref) > max_co:
                max_co = corref
                best_feature = i
        return best_feature,max_co
    
    def step_wise(self,data,n):
        from sklearn.linear_model import LinearRegression
        from pandas import DataFrame as df
        data = data.fillna(-1)
        y = df(data['rule104'])
        X = data.drop('rule104',axis=1)
        residual = df(data['rule104'])
        residual.columns = ['residual']
        best_fea = []
        abs_corrcoef = []
        feature_opt = list(X.columns.values)
    
        for i in range(n):
            a,b = self.find_best_feature(X,feature_opt,residual)
            
            best_fea.append(a)
            abs_corrcoef.append(b)
            feature_opt.remove(a)
            print(best_fea)
            print(feature_opt)
            ls = LinearRegression().fit(df(X[best_fea]), y)

            y_pre = df(ls.predict(df(X[best_fea])))
            y_pre.columns = ['y_pre']
            residual['residual'] = y['rule104'] - y_pre['y_pre']
            
        return best_fea,abs_corrcoef

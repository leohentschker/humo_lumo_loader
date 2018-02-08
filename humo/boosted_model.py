from model_wrapper import SKLearnModelWrapper
import xgboost as xbg

class BoostRegression(SKLearnModelWrapper):
    def get_model(self):
        return xgboost.XGBRegressor(n_estimators=100, max_depth=7)

boosted = BoostRegression("coulomb_huge.csv.csv")
boosted.build_model()
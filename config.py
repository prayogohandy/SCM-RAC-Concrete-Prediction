# Features
feature_bounds = {
    'C': (0.0, 800.0),
    'W': (50.0, 350.0),
    'CA': (0.0, 1800.0),
    'RCA': (0.0, 1500.0),
    'FAg': (0.0, 1600.0),
    'FA': (0.0, 420.0),
    'SF': (0.0, 300.0),
    'GGBFS': (0.0, 450.0),
    'CC': (0.0, 300.0),
    'SP': (0.0, 50.0),
}

default_values = {
    'C': 350.0,
    'W': 175.0,
    'CA': 750.0,
    'RCA': 0.0,
    'FAg': 1050.0,
    'FA': 0.0,
    'SF': 0.0,
    'GGBFS': 0.0,
    'CC': 0.0,
    'SP': 3.5,
}

feature_steps = {
    'C': 1.0,
    'W': 1.0,
    'CA': 1.0,
    'RCA': 1.0,
    'FAg': 1.0,
    'FA': 0.1,
    'SF': 0.1,
    'GGBFS': 0.1,
    'CC': 0.1,
    'SP': 0.01,
}

model_folder = "model"

model_abbreviations = {
        "RandomForestRegressor": "RF",
        "ExtraTreesRegressor": "ET",
        "GradientBoostingRegressor": "GB",
        "DecisionTreeRegressor": "DT",
        "KNeighborsRegressor": "KNN",
        "SVR": "SVR",
        "XGBRegressor": "XGB",
        "LGBMRegressor": "LGB",
        "CatBoostRegressor": "CB",
        "MLPRegressor": "MLP",
        "AdaBoostRegressor": "ADA",
        "HistGradientBoostingRegressor": "HGB",
        "GaussianProcessRegressor": "GPR",
        "Ridge": "Ridge",
    }

model_abbreviations_lower = {k.lower(): v for k, v in model_abbreviations.items()}

model_param_keys = {
    "RandomForestRegressor": ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_features", "bootstrap"],
    "ExtraTreesRegressor": ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_features", "bootstrap"],
    "GradientBoostingRegressor": ["n_estimators", "learning_rate", "max_depth", "min_samples_split", "min_samples_leaf", "max_features", "subsample"],
    "KNeighborsRegressor": ["n_neighbors", "weights", "p"],
    "SVR": ["C", "epsilon", "gamma", "kernel", "degree"],  # degree only if kernel="poly"
    "XGBRegressor": ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda", "gamma", "min_child_weight"],
    "LGBMRegressor": ["n_estimators", "max_depth", "learning_rate", "num_leaves", "min_child_samples", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda", "min_split_gain", "verbose"],
    "CatBoostRegressor": ["iterations", "depth", "learning_rate", "l2_leaf_reg", "border_count", "random_strength", "bagging_temperature", "verbose", "early_stopping_rounds"],
    "MLPRegressor": ["n_layers", "hidden_layer_sizes", "activation", "solver", "alpha", "learning_rate_init", "max_iter", "early_stopping"],
    "AdaBoostRegressor": ["n_estimators", "learning_rate", "loss"],
    "HistGradientBoostingRegressor": ["max_iter", "max_leaf_nodes", "learning_rate", "max_depth", "min_samples_leaf", "l2_regularization", "early_stopping"],
    "Ridge": ["alpha", "fit_intercept"],
}

rename_input = {
    # Input features
    "C": "Cement (kg/m³)",
    "W": "Water (kg/m³)",
    "CA": "Coarse Aggregate (kg/m³)",
    "RCA": "Recycled Coarse Aggregate (kg/m³)",
    "FAg": "Fine Aggregate (kg/m³)",
    "SP": "Superplasticizer (kg/m³)",
    "FA": "Fly Ash (kg/m³)",
    "SF": "Silica Fume (kg/m³)",
    "GGBFS": "Blast Furnace Slag (kg/m³)",
    "CC": "Calcined Clay (kg/m³)"
}

reverse_rename_input = {v: k for k, v in rename_input.items()}

rename_dict = {
    # Input features
    "C": r'$\mathrm{Cement}$ [kg/m$^3$]',
    "W": r'$\mathrm{Water}$ [kg/m$^3$]',
    "CA": r'$\mathrm{Coarse\ Aggregate}$ [kg/m$^3$]',
    "RCA": r'$\mathrm{Recycled\ Coarse\ Aggregate}$ [kg/m$^3$]',
    "FAg": r'$\mathrm{Fine\ Aggregate}$ [kg/m$^3$]',
    "SP": r'$\mathrm{Superplasticizer}$ [kg/m$^3$]',
    "FA": r'$\mathrm{Fly\ Ash}$ [kg/m$^3$]',
    "SF": r'$\mathrm{Silica\ Fume}$ [kg/m$^3$]',
    "GGBFS": r'$\mathrm{Ground\ Granulated\ Blast\ Furnace\ Slag}$ [kg/m$^3$]',
    "CC": r'$\mathrm{Calcined\ Clay}$ [kg/m$^3$]',

    # Binder-related derived features
    "SCM Content": r'$\mathrm{SCM}$ [kg/m$^3$]',
    "Binder Content": r'$\mathrm{Binder\ Content}$ [kg/m$^3$]',
    "W/B Ratio": r'$W/B$ [-]',
    "SCM RR": r'$RR_{\mathrm{SCM}}$ [-]',
    "Cement%": r'$Cement\%$ [-]',
    "FA%": r'$FA\%$ [-]',
    "SF%": r'$SF\%$ [-]',
    "GGBFS%": r'$GGBFS\%$ [-]',
    "CC%": r'$CC\%$ [-]',
    # Aggregate-related derived features
    "Total Aggregate Content (kg/m3)": r'$\mathrm{Aggregate\ Content}$ [kg/m$^3$]',
    "Coarse%": r'$Coarse\%$ [-]',
    "RCA RR": r'$RR_{\mathrm{RCA}}$ [-]',

    # Additional mix design ratios
    "Sp/B Ratio": r'$SP/B$ [-]',
    "Sp/W Ratio": r'$SP/W$ [-]',
    "P/Agg Ratio": r'$P/A$ [-]',
    "B/Agg Ratio": r'$B/A$ [-]',
    "W/S Ratio": r'$W/S$ [-]',

    # Outputs
    "CS":  "Cylinder compressive strength (MPa)",
    "TS":  "Splitting tensile strength (MPa)",
    "FS":  "Flexural strength (MPa)",
    "E":   "Elastic modulus (GPa)",
    "SL":  "Slump (cm)",
    "CP":  "Chloride permeability (Coulomb)",
    "DS":  "Drying shrinkage (με)",
}

reverse_rename_dict = {v: k for k, v in rename_dict.items()}

OUTPUT_FORMAT = {
    "CS": {"unit": "MPa", "decimals": 2},
    "TS": {"unit": "MPa", "decimals": 2},
    "FS": {"unit": "MPa", "decimals": 2},
    "E":  {"unit": "GPa", "decimals": 2},
    "SL": {"unit": "cm",  "decimals": 1},
    "DS": {"unit": "µε",  "decimals": 0},
    "CP": {"unit": "C",   "decimals": 0},
}


input_cols = ['Cement (kg/m3)', 'Water (kg/m3)', 'Coarse aggregate (kg/m3)',
            'Recycled coarse aggregate (kg/m3)', 'Fine aggregate (kg/m3)', 
            'FA (kg/m3)', 'SF (kg/m3)', 'GGBFS (kg/m3)', 'CC (kg/m3)', 'SP (kg/m3)']

input_cols_fe = ['SCM Content', 'Binder Content', 'W/B Ratio', 'SCM RR', 'Cement%', 
            'FA%', 'SF%', 'GGBFS%', 'CC%', 
            'Coarse%', 'Total Aggregate Content (kg/m3)', 'RCA RR',
            'Sp/B Ratio', 'Sp/W Ratio', 'P/Agg Ratio', 'B/Agg Ratio', 'W/S Ratio']

input_cols_all = input_cols + input_cols_fe
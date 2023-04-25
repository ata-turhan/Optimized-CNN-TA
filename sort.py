def model_ho(model_name, datas, parameter_space):
    set_random_seed()
    start_time = time.time()
    model_creations = {
        "MLP": create_model_MLP,
        "LSTM": create_model_LSTM,
        "GRU": create_model_GRU,
        "CNN-2D": create_model_CNN_2D,
    }

    def objective(trial):
        activation_func = trial.suggest_categorical(
            name="activation_func", choices=parameter_space["activation_func"]
        )
        dropout_rate = trial.suggest_categorical(
            "dropout_rate", parameter_space["dropout_rate"]
        )
        optimizer_algo = trial.suggest_categorical(
            "optimizer_algo", parameter_space["optimizer_algo"]
        )
        batch_size = trial.suggest_categorical(
            "batch_size", parameter_space["batch_size"]
        )
        # lr_max = trial.suggest_categorical("learning_rate_max", [1e-1, 1e-2, 1e-3, 1e-4])

        create_model = model_creations[model_name]
        model = create_model_MLP(activation_func, dropout_rate, optimizer_algo)

        f1_scores = []

        for i in range(len(datas)):
            OUTPUT_PATH = "./outputs"
            es = EarlyStopping(
                monitor="f1_score", mode="max", verbose=1, patience=30, min_delta=1e-3
            )
            mcp = ModelCheckpoint(
                os.path.join(OUTPUT_PATH, f"best_CNN_model-{i+1}.h5"),
                monitor="f1_score",
                verbose=0,
                save_best_only=True,
                save_weights_only=False,
                mode="max",
            )

            if model_name == "MLP":
                val_split_point = int(0.8 * len(datas[i][0]))
                X_train = datas[i][0][:val_split_point].iloc[:, :-1]
                y_train = tf.keras.utils.to_categorical(
                    datas[i][0][:val_split_point].iloc[:, -1], num_classes=3
                )
                X_val = datas[i][0][val_split_point:].iloc[:, :-1]
                y_val = datas[i][0][val_split_point:].iloc[:, -1]
            else:
                val_split_point = int(0.8 * len(datas[0][i]))
                X_train = datas[0][i][:val_split_point]
                y_train = datas[1][i][:val_split_point]
                X_val = datas[0][i][val_split_point:]
                y_val = datas[1][i][val_split_point:]
                y_val = y_val.argmax(axis=-1)

            model = create_model(activation_func, dropout_rate, optimizer_algo)
            model.fit(
                X_train,
                y_train,
                batch_size=batch_size,
                epochs=5,
                verbose=0,
                callbacks=[es, mcp],
                class_weight={0: 1, 1: 10, 2: 10},
            )
            y_pred = model.predict(X_val)
            y_pred = y_pred.argmax(axis=-1)
            f1_scores.append(f1_score(y_val, y_pred, average="macro"))
        return np.mean(f1_scores)

    study = optuna.create_study(
        study_name=f"{model_name}_Bayesian_Optimization",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=5)
    trial = study.best_trial

    print("\n------------------------------------------")
    print("Best F1 Macro: {}".format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))
    minutes = round(int(time.time() - start_time) / 60, 2)
    print(f"\nCompleted in {minutes} minutes")
    return trial.params
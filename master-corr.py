import pandas as pd

dataset = 'mnist'

# Value 1
activator = "sgmd"
df_cnn_relu0_1 = pd.read_csv(dataset + "/results/" + activator + "/cnn_K0.csv")
df_e_1 = pd.read_csv(dataset + "/results/" + activator + "/cnn_test_class_loss.csv")
df_error_1 = pd.DataFrame(columns=["0", "1", "2", "3", "4"])
df_error_1["0"] = df_e_1["0"]
df_error_1["1"] = df_e_1["0"]
df_error_1["2"] = df_e_1["0"]
df_error_1["3"] = df_e_1["0"]
df_error_1["4"] = df_e_1["0"]
activator = "tanh"
df_cnn_relu0_2 = pd.read_csv(dataset + "/results/" + activator + "/cnn_K0.csv")
df_e_2 = pd.read_csv(dataset + "/results/" + activator + "/cnn_test_class_loss.csv")
df_error_2 = pd.DataFrame(columns=["0", "1", "2", "3", "4"])
df_error_2["0"] = df_e_2["0"]
df_error_2["1"] = df_e_2["0"]
df_error_2["2"] = df_e_2["0"]
df_error_2["3"] = df_e_2["0"]
df_error_2["4"] = df_e_2["0"]
activator = "relu"
df_cnn_relu0_3 = pd.read_csv(dataset + "/results/" + activator + "/cnn_K0.csv")
df_e_3 = pd.read_csv(dataset + "/results/" + activator + "/cnn_test_class_loss.csv")
df_error_3 = pd.DataFrame(columns=["0", "1", "2", "3", "4"])
df_error_3["0"] = df_e_3["0"]
df_error_3["1"] = df_e_3["0"]
df_error_3["2"] = df_e_3["0"]
df_error_3["3"] = df_e_3["0"]
df_error_3["4"] = df_e_3["0"]

sgmd_k0_1 = df_cnn_relu0_1.corrwith(df_error_1)
tanh_k0_1 = df_cnn_relu0_2.corrwith(df_error_2)
relu_k0_1 = df_cnn_relu0_3.corrwith(df_error_3)

# Value 2
activator = "sgmd"
df_cnn_relu0_1 = pd.read_csv(dataset + "/results/" + activator + "/cnn_K1.csv")
activator = "tanh"
df_cnn_relu0_2 = pd.read_csv(dataset + "/results/" + activator + "/cnn_K1.csv")
activator = "relu"
df_cnn_relu0_3 = pd.read_csv(dataset + "/results/" + activator + "/cnn_K1.csv")

sgmd_k1_1 = df_cnn_relu0_1.corrwith(df_error_1)
tanh_k1_1 = df_cnn_relu0_2.corrwith(df_error_2)
relu_k1_1 = df_cnn_relu0_3.corrwith(df_error_3)


# Value 3
activator = "sgmd"
df_cnn_relu0_1 = pd.read_csv(dataset + "/results/" + activator + "/cnn_K2.csv")
activator = "tanh"
df_cnn_relu0_2 = pd.read_csv(dataset + "/results/" + activator + "/cnn_K2.csv")
activator = "relu"
df_cnn_relu0_3 = pd.read_csv(dataset + "/results/" + activator + "/cnn_K2.csv")

sgmd_k2_1 = df_cnn_relu0_1.corrwith(df_error_1)
tanh_k2_1 = df_cnn_relu0_2.corrwith(df_error_2)
relu_k2_1 = df_cnn_relu0_3.corrwith(df_error_3)


# Value 4
activator = "sgmd"
df_cnn_relu0_1 = pd.read_csv(dataset + "/results/" + activator + "/cnn_K3.csv")
activator = "tanh"
df_cnn_relu0_2 = pd.read_csv(dataset + "/results/" + activator + "/cnn_K3.csv")
activator = "relu"
df_cnn_relu0_3 = pd.read_csv(dataset + "/results/" + activator + "/cnn_K3.csv")

sgmd_k3_1 = df_cnn_relu0_1.corrwith(df_error_1)
tanh_k3_1 = df_cnn_relu0_2.corrwith(df_error_2)
relu_k3_1 = df_cnn_relu0_3.corrwith(df_error_3)


# Value 5
activator = "sgmd"
df_cnn_relu0_1 = pd.read_csv(dataset + "/results/" + activator + "/cnn_K4.csv")
activator = "tanh"
df_cnn_relu0_2 = pd.read_csv(dataset + "/results/" + activator + "/cnn_K4.csv")
activator = "relu"
df_cnn_relu0_3 = pd.read_csv(dataset + "/results/" + activator + "/cnn_K4.csv")

sgmd_k4_1 = df_cnn_relu0_1.corrwith(df_error_1)
tanh_k4_1 = df_cnn_relu0_2.corrwith(df_error_2)
relu_k4_1 = df_cnn_relu0_3.corrwith(df_error_3)



###############################################################
# Value 1
activator = "sgmd"
df_cnn_relu0_1 = pd.read_csv(dataset + "/results/" + activator + "/cnn_K5.csv")
activator = "tanh"
df_cnn_relu0_2 = pd.read_csv(dataset + "/results/" + activator + "/cnn_K5.csv")
activator = "relu"
df_cnn_relu0_3 = pd.read_csv(dataset + "/results/" + activator + "/cnn_K5.csv")

sgmd_k5_1 = df_cnn_relu0_1.corrwith(df_error_1)
tanh_k5_1 = df_cnn_relu0_2.corrwith(df_error_2)
relu_k5_1 = df_cnn_relu0_3.corrwith(df_error_3)


# Value 2
activator = "sgmd"
df_cnn_relu0_1 = pd.read_csv(dataset + "/results/" + activator + "/cnn_K6.csv")
activator = "tanh"
df_cnn_relu0_2 = pd.read_csv(dataset + "/results/" + activator + "/cnn_K6.csv")
activator = "relu"
df_cnn_relu0_3 = pd.read_csv(dataset + "/results/" + activator + "/cnn_K6.csv")

sgmd_k6_1 = df_cnn_relu0_1.corrwith(df_error_1)
tanh_k6_1 = df_cnn_relu0_2.corrwith(df_error_2)
relu_k6_1 = df_cnn_relu0_3.corrwith(df_error_3)


# Value 3
activator = "sgmd"
df_cnn_relu0_1 = pd.read_csv(dataset + "/results/" + activator + "/cnn_K7.csv")
activator = "tanh"
df_cnn_relu0_2 = pd.read_csv(dataset + "/results/" + activator + "/cnn_K7.csv")
activator = "relu"
df_cnn_relu0_3 = pd.read_csv(dataset + "/results/" + activator + "/cnn_K7.csv")

sgmd_k7_1 = df_cnn_relu0_1.corrwith(df_error_1)
tanh_k7_1 = df_cnn_relu0_2.corrwith(df_error_2)
relu_k7_1 = df_cnn_relu0_3.corrwith(df_error_3)

# Value 4
activator = "sgmd"
df_cnn_relu0_1 = pd.read_csv(dataset + "/results/" + activator + "/cnn_K8.csv")
activator = "tanh"
df_cnn_relu0_2 = pd.read_csv(dataset + "/results/" + activator + "/cnn_K8.csv")
activator = "relu"
df_cnn_relu0_3 = pd.read_csv(dataset + "/results/" + activator + "/cnn_K8.csv")

sgmd_k8_1 = df_cnn_relu0_1.corrwith(df_error_1)
tanh_k8_1 = df_cnn_relu0_2.corrwith(df_error_2)
relu_k8_1 = df_cnn_relu0_3.corrwith(df_error_3)

# Value 5
activator = "sgmd"
df_cnn_relu0_1 = pd.read_csv(dataset + "/results/" + activator + "/cnn_K9.csv")
activator = "tanh"
df_cnn_relu0_2 = pd.read_csv(dataset + "/results/" + activator + "/cnn_K9.csv")
activator = "relu"
df_cnn_relu0_3 = pd.read_csv(dataset + "/results/" + activator + "/cnn_K9.csv")

sgmd_k9_1 = df_cnn_relu0_1.corrwith(df_error_1)
tanh_k9_1 = df_cnn_relu0_2.corrwith(df_error_2)
relu_k9_1 = df_cnn_relu0_3.corrwith(df_error_3)


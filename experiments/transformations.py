import pandas as pd


def main():

    pictures = '../result_pictures'

    df_all_models = pd.read_csv(open("../results/all_models_full.csv"))
    print(df_all_models)
    attacks = list(df_all_models.attack.unique())
    for attack in attacks:
        df_specific_attack = df_all_models.loc[df_all_models['attack'] == attack]

        df_specific_attack.sort_values(by=['iteration']).plot(x='iteration', y='accuracy').figure.savefig(
            f"{pictures}/{attack}_accuracy_by_iterations.png")

        df_specific_attack.sort_values(by=['attack_time']).plot(x='attack_time', y='accuracy').figure.savefig(
            f"{pictures}/{attack}_accuracy_by_attack_time.png")

        df_specific_attack.sort_values(by=['iteration']).plot(x='iteration', y='loss').figure.savefig(
            f"{pictures}/{attack}_loss_by_iterations.png")

        df_specific_attack.sort_values(by=['attack_time']).plot(x='attack_time', y='loss').figure.savefig(
            f"{pictures}/{attack}_loss_by_attack_time.png")





if __name__ == '__main__':
    main()

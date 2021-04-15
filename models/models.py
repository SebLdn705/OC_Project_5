import pandas as pd

def prediction(model,
               threshold: float,
               text,
               target_col: list):

    '''

    Used in order to predict the StackOverflow tags

    :param model: SKlearn model saved to be used for the prediction
    :param threshold: threshold to be used for acceptance of a positve tag
    :param text: question to be used for the prediction
    :param target_col: column to be used. They represent the tags searched
    :return: dictionary containing the different tags suggested
    '''

    Y_pred_proba = model.predict_proba(text)
    tmp_results = []
    for n_proba in range(len(Y_pred_proba)):

        for i in range(len(Y_pred_proba[n_proba])):
            if Y_pred_proba[n_proba][i][1] > threshold:
                tmp_results.append(1)
            else:
                tmp_results.append(0)

    target_col = [x.replace('target_', '') for x in target_col]

    d = {'tag': target_col, 'flag': tmp_results}
    df_predict = pd.DataFrame.from_dict(d, orient='columns')
    mask = df_predict.flag == 1
    df_out = df_predict.loc[mask, ['tag']]

    return df_out.to_dict(orient='list')


def prediction_word_2_vec(word2vec,
                          model,
                          threshold,
                          target_col: list):

    pass
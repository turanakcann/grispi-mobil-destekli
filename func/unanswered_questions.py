import pandas as pd
from read_json import dosyaOkuma

df = dosyaOkuma('data.json')

user_messages = df[df['sender_id'].isna() | df['sender_id'].str.contains('user_id_pattern')]

def is_unanswered(row, df):
    next_idx = df.index[df.index > row.name].min()
    if pd.isna(next_idx):
        return True
    next_msg = df.loc[next_idx]
    return next_msg['sender_id'] != 'bf17272dc3f0'

user_messages['unanswered'] = user_messages.apply(lambda x: is_unanswered(x, df), axis=1)
unanswered_questions = user_messages[user_messages['unanswered']]

print(unanswered_questions[['content.text', 'created_at']])

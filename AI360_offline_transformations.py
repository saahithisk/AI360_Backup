# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

############################################
# Set up
############################################

import os
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
import fsspec
#from google.cloud import storage
#import gcsfs
#import authenticate

############################################
# Transform data
############################################

def transform_qualtrics():

    global all_questions_text
    global question_enum_mapping
    global mapping_path
    global today_str
    global yesterday_str
    
    bucket_name = 'gs://snowflake-preload-qualtrics'
    mapping_folder = 'qualtrics_mappings'
    mapping_path = os.path.join(bucket_name, mapping_folder)
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    today_str = today.strftime("_%m_%d_%y")
    yesterday_str = yesterday.strftime("_%m_%d_%y")

    # Read in raw data
    #raw = ctx.execute(f'SELECT * FROM {raw_data};').fetch_pandas_all()
    
    
    raw = pd.read_csv("C:\\Users\\ungwe\\Downloads\\surveys_AI360_Readiness_Assessment_03_16_23.csv")


    # Drop metadata and unnecessary columns
    df = raw.drop(axis=0, labels=[0]).reset_index(drop=True)
    drop_cols = ['STARTDATE',
                'ENDDATE',
                'STATUS',
                'IPADDRESS',
                'PROGRESS',
                'DURATION (IN SECONDS)',
                'FINISHED',
                'RECORDEDDATE',
                'RECIPIENTLASTNAME',
                'RECIPIENTFIRSTNAME',
                'RECIPIENTEMAIL',
                'LOCATIONLATITUDE',
                'LOCATIONLONGITUDE',
                'DISTRIBUTIONCHANNEL',
                'USERLANGUAGE',
                'INTRO',
                'Q_DATAPOLICYVIOLATIONS']
    df.drop(columns=drop_cols, inplace=True)
    df = df.rename(columns={'RESPONSEID': 'SUBMISSION_ID'})
    submission_ids=df.SUBMISSION_ID.iloc[1:].values.tolist()
    df = df.T.reset_index(drop=True)

    # update column names for readability
    df.rename(columns={0: 'question_id'}, inplace=True)
    sub_cols = {x: 'submission_' + str(x-1) for x in df.columns[1:].values.tolist()}
    df.rename(columns=sub_cols, inplace=True)

    # Update question ids
    df.question_id = df.question_id.apply(lambda x: x.replace('"','').replace('{', '').replace('}', '').replace('ImportId:', '').replace('_', ''))

    # Get answers in one column
    df = pd.wide_to_long(df, stubnames='submission_', i='question_id', j='submission_id').sort_index(level=0).reset_index()
    mapped_submissions = dict(zip(np.arange(int(df.submission_id.value_counts().index[-1])+1), submission_ids))
    df.submission_id = df.submission_id.map(mapped_submissions)
    df.rename(columns={'submission_': 'answer'}, inplace=True)

    # Drop recordID rows
    record_index = df[df.question_id == 'recordId'].index.values[0]
    df = df.drop(axis=0, labels=df[record_index:].index.tolist()).reset_index(drop=True)

    # Read in question details
    filepath = 'gs://snowflake-preload-qualtrics/survey_details/survey_details.json'
    with fsspec.open(filepath, mode='r') as f:
        questions = pd.read_json(f, orient='records').T[['questions']]
        questions = pd.json_normalize(questions.to_dict(orient='records'))

    # Get question text
    all_questions_text = questions.filter(like='questionText').T.reset_index().rename(columns={'index': 'question_id_details', 0: 'wording'}).drop(columns=[1])
    all_questions_text.question_id_details = all_questions_text.question_id_details.apply(lambda x: x.replace('questions.', '').replace('.questionText', ''))
    all_questions_text.wording = all_questions_text.wording.str.replace('<.*?>', '', regex=True)
    all_questions_text['question_id_short'] = all_questions_text.wording.apply(lambda x: re.search('[A-Za-z0-9]+', x).group(0))
    all_questions_text.wording = all_questions_text.wording.str.replace('[A-Za-z0-9]+\.', '', regex=True)

    # Merge back into df
    df['question_id_details'] = df.question_id.apply(lambda x: x[:5] if x != 'externalDataReference' else x)
    df = df.merge(all_questions_text, how='left', on='question_id_details')

    # Map question types
    filename = f'grouping_and_question_type{yesterday_str}.csv'
    filepath = os.path.join(mapping_path, filename)
    with fsspec.open(filepath, mode='r') as f:
        grouping_and_question_type_mapping = pd.read_csv(f)

    question_types = grouping_and_question_type_mapping.groupby('question_type')['question_id'].apply(lambda x: list(x)).reset_index()
    question_type_mapping = dict(zip(question_types.question_type, question_types.question_id))
    df['question_type'] = df.question_id.map({item: k for k, v in question_type_mapping.items() for item in v})

    # Merge on df to capture TEXT questions
    answer_mapping = questions.filter(like='choiceText').T.reset_index().rename(columns={'index': 'question_id_choice', 0: 'text_answer'}).drop(columns=[1])

    # Get just answer choices
    mask = (~answer_mapping.question_id_choice.str.contains('subQuestions'))
    answer_mapping = answer_mapping.loc[mask]

    # Clean
    answer_mapping.question_id_choice = answer_mapping.question_id_choice.str.replace('[a-zA-Z]+\.', '', regex=True).replace('[a-zA-Z]+\.', regex=True)
    answer_mapping.question_id_choice = answer_mapping.question_id_choice.str.replace('.choiceText', '').replace('.', '')
    answer_mapping.question_id_choice = answer_mapping.question_id_choice.str.replace('.', '')
    answer_mapping['question_id_details'] = answer_mapping.question_id_choice.apply(lambda x: x[:5])
    answer_mapping['answer'] = answer_mapping.question_id_choice.apply(lambda x: x[-1] if len(x) <= 6 else x[-2:]) # When questions have more than 9 choices

    # Merge on df to capture TEXT questions
    answer_mapping = answer_mapping.merge(df[['question_id_details', 'question_id', 'question_type']], on='question_id_details', how='left')
    answer_mapping = answer_mapping.groupby(['question_id', 'question_id_details', 'text_answer', 'question_type'])['answer'].apply(lambda x: list(x)[0]).reset_index()

    # Update questions with text inputs
    answer_mapping.answer = np.where(answer_mapping.question_id.str.contains('TEXT'), answer_mapping.question_id.str[-5], answer_mapping.answer)
    answer_mapping.text_answer = np.where(answer_mapping.question_id.str.contains('TEXT'), np.nan, answer_mapping.text_answer)

    # Update questions with multi-punch
    answer_mapping.question_id = np.where((answer_mapping.question_type == 'Multi-Punch') & ~(answer_mapping.question_id.str.contains("TEXT")) & (answer_mapping.question_id.str[-2] == ':'), answer_mapping.question_id.str[:-1] + answer_mapping.answer, answer_mapping.question_id)
    answer_mapping.question_id = np.where((answer_mapping.question_type == 'Multi-Punch') & ~(answer_mapping.question_id.str.contains("TEXT")) & (answer_mapping.question_id.str[-3] == ':'), answer_mapping.question_id.str[:-2] + answer_mapping.answer, answer_mapping.question_id)
    answer_mapping.answer = np.where(answer_mapping.question_type == 'Multi-Punch', 1, answer_mapping.answer)
    answer_mapping = answer_mapping.drop_duplicates(subset=['question_id', 'text_answer', 'answer']).reset_index(drop=True)

    # Use answer mapping to create enumerated question id column.
    question_enum_mapping = answer_mapping.drop_duplicates('question_id')[['question_id', 'question_id_details']].reset_index(drop=True)
    question_enum_mapping = question_enum_mapping.merge(all_questions_text[['question_id_details', 'question_id_short']], on='question_id_details', how='left')

    mask = (~question_enum_mapping.question_id.isin(['QID10TEXT', 'QID11TEXT', 'QID12TEXT'])) & (question_enum_mapping.question_id.str.len() > 5) & (~question_enum_mapping.question_id.str.contains('choiceID', na=False))
    question_enum_mapping = question_enum_mapping.loc[mask]# .dropna()
    question_enum_mapping

    # Probably could turn this into a mapping too
    def enumerate_questions(x):

        """
        Use question ids and short questions ids to create enumerated question id column.
        """

        try:
            # All subquestions that aren't the text box input for "Other"
            if not 'TEXT' in x['question_id']:
                if len(x['question_id']) > 6:
                    if 'choiceId:' in x['question_id']: # questions with choiceId in them
                        x = x['question_id_short'] + '_' + x['question_id'].split(':')[1]
                    elif 'x' in x['question_id']: # questions with 'x' in them
                        x = x['question_id_short'] + '_' + x['question_id'][-1]
                    else: 
                        x = x['question_id_short'] + '_' + x['question_id'][-2:] # More than 9 subquestions
                else:
                    x = x['question_id_short'] + '_' + x['question_id'][-1]
                    
            else:
                # Subquestions that are "Other" where "Other" isn't the 10th option
                if len(x['question_id']) == 10:
                    x = x['question_id_short'] + '_' + x['question_id'][-5] + '_TEXT'

                # # More than 9 subquestions
                elif len(x['question_id']) > 10:
                    if 'x7TEXT' in x['question_id']: # Weird questions containing 'x' in the question ID
                        x = x['question_id_short'] + '_' + x['question_id'][-5:-4] + '_TEXT'
                    else:    
                        x = x['question_id_short'] + '_' + x['question_id'][-6:-4] + '_TEXT'
                # Questions that are just text inputs
                else:
                    x = x['question_id_short'] + '_TEXT'
        except: # NaNs
            return x
        return x

    # Create enumerated question ids
    question_enum_mapping['question_id_enum'] = question_enum_mapping.apply(lambda x: enumerate_questions(x), axis=1)
    df = df.merge(question_enum_mapping[['question_id', 'question_id_enum']], how='left', on='question_id')
    df.question_id_enum = np.where(df.question_id_enum.isna(), df.question_id_short, df.question_id_enum)

    # Map groupings
    groupings = grouping_and_question_type_mapping.groupby('grouping')['question_id'].apply(lambda x: list(x)).reset_index()
    groupings_mapping = dict(zip(groupings.grouping, groupings.question_id))
    df['grouping'] = df.question_id.map({item: k for k, v in groupings_mapping.items() for item in v})

    # Map text answers
    filename = f'text_answer_mapping{yesterday_str}.csv'
    filepath = os.path.join(mapping_path, filename)
    with fsspec.open(filepath, mode='r') as f:
        text_answers = pd.read_csv(f)
    text_answers = text_answers.dropna()
    text_answers.answer = text_answers.answer.astype(int).astype(str)
    df = df.join(text_answers[['question_id', 'answer', 'text_answer']].set_index(['question_id', 'answer']), on=['question_id', 'answer'])
    df.text_answer = np.where(df.text_answer.isna(), df.answer, df.text_answer)

    # Map likert values
    filename = f'likert_mapping{yesterday_str}.csv'
    filepath = os.path.join(bucket_name, mapping_folder, filename)
    with fsspec.open(filepath, mode='r') as f:
        values = pd.read_csv(f)
    values = values.groupby('value')['text_answer'].apply(lambda x: list(x)).reset_index()
    value_mapping = dict(zip(values.value, values.text_answer))
    df['value'] = df.text_answer.map({item: k for k, v in value_mapping.items() for item in v})

    # Add demographic persistence
    # Recreate demographic_question_ids and omit free-text answer for QID13 for other and only persistent "Other" value, remove QID14
    demographic_question_ids = ['externalDataReference', 'QID11TEXT', 'QID12TEXT', 'QID13']
    demographics_df = df[df['question_id'].isin(demographic_question_ids)].reset_index(drop=True)
    demographics_df = demographics_df[['submission_id', 'question_id', 'question_id_details', 'answer']]
    demographics_df.pivot(index='submission_id', columns='question_id', values='answer').reset_index()

    # Create pivot table of demographics
    demographics_df_pivot_table = demographics_df.pivot(index='submission_id', columns='question_id', values='answer').reset_index().rename(columns={'externalDataReference': 'organization', 'QID11TEXT': 'division', 'QID12TEXT': 'title', 'QID13': 'level'})
    demographics_df_pivot_table.columns.name = None

    # Map levels
    levels = {
        '1': 'Executive / C-Suite / Director',
        '2': 'Management / Associate',
        '3': 'Staff',
        '4': 'Other'
    }
    demographics_df_pivot_table.level = demographics_df_pivot_table.level.map(levels, na_action='ignore')

    # Merge back into main df
    df = df.merge(demographics_df_pivot_table)

    # Drop externalDataReference rows
    df = df.drop(df[df.question_id == 'externalDataReference'].index).reset_index(drop=True)

    # Reorder columns
    df_final = df[[
        'submission_id',
        'organization',
        'division',
        'title',
        'level',
        'question_id',
        'question_id_details',
        'question_id_short',
        'question_id_enum',
        'wording',
        'question_type',
        'grouping',
        'answer',
        'text_answer',
        'value'
    ]]
    df_final

    df_final['Q_ID'] = df_final['question_id_enum']
    df_final['State'] = "COMPLETED"

    return df_final

def write_mappings(df):

    grouping_and_question_type_name = 'grouping_and_question_type'
    text_answer_mapping_name = 'text_answer_mapping'
    likert_mapping_name = 'likert_mapping'

    # Get one-to-one relationships (grouping and question_type)
    df_for_mapping = df[['question_id', 'question_id_details', 'question_id_short', 'question_id_enum', 'wording', 'question_type', 'grouping']].drop_duplicates()

    # Get all survey questions
    one_to_one_mapping = all_questions_text[all_questions_text.question_id_details.isin(df.question_id_details)].reset_index(drop=True)
    one_to_one_mapping = one_to_one_mapping.merge(question_enum_mapping, on=['question_id_details', 'question_id_short'], how='left')

    # Include any survey questions that are not included in submission
    # This should theoretically give all the questions accounted for by the submission and the survey since these may not always overlap.
    one_to_one_mapping = one_to_one_mapping.merge(df_for_mapping, on=['question_id_details', 'question_id_enum', 'question_id', 'question_id_short', 'wording'], how='right')

    # Compare to yesterday's pull and save
    filepath = os.path.join(mapping_path, f'{grouping_and_question_type_name}{yesterday_str}.csv')
    with fsspec.open(filepath, mode='r') as f:
        grouping_and_question_type = pd.read_csv(f)

    grouping_and_question_type = grouping_and_question_type.merge(one_to_one_mapping, how='right', on=['question_id_details', 'question_id_enum', 'question_id', 'question_id_short', 'wording', 'question_type', 'grouping'])

    filepath = os.path.join(mapping_path, f'{grouping_and_question_type_name}{today_str}.csv')
    with fsspec.open(filepath, mode='w') as f:
        grouping_and_question_type.to_csv(f)
    
    # Text answers have a one-to-many relationship with questions. Use one_to_one_mapping since it should be an exhaustive list of all survey questions.
    df_one_to_many = one_to_one_mapping[['question_id', 'question_id_details', 'question_id_short', 'question_id_enum']].drop_duplicates('question_id')

    # Compare to yesterday's pull and save
    filepath = os.path.join(mapping_path, f'{text_answer_mapping_name}{yesterday_str}.csv')
    with fsspec.open(filepath, mode='r') as f:
        one_to_many_mapping = pd.read_csv(f)
    one_to_many_mapping = one_to_many_mapping.merge(df_one_to_many, on=['question_id_details', 'question_id_enum', 'question_id', 'question_id_short'], how='right')

    filepath = os.path.join(mapping_path, f'{text_answer_mapping_name}{today_str}.csv')
    with fsspec.open(filepath, mode='w') as f:
        one_to_many_mapping.to_csv(f)

    # Likert values. Use one_to_one_mapping since it should be an exhaustive list of all survey questions.
    likert_questions = one_to_one_mapping[one_to_one_mapping.question_type == 'Likert']
    likert_questions = likert_questions.merge(one_to_many_mapping, how='left', on=['question_id_details', 'question_id_enum', 'question_id', 'question_id_short'])[['question_id_details', 'question_id_enum', 'question_id', 'question_id_short', 'text_answer']].drop_duplicates('text_answer').reset_index(drop=True)

    # Merge back on df to get all the values mapped to the text_answers
    df_for_likert_mapping = df[['text_answer', 'value']][df.question_type == 'Likert'].drop_duplicates().reset_index(drop=True)
    likert_questions = likert_questions.merge(df_for_likert_mapping, on='text_answer', how='left')

    # Compare to yesterday's pull and save
    filepath = os.path.join(mapping_path, f'{likert_mapping_name}{yesterday_str}.csv')
    with fsspec.open(filepath, mode='r') as f:
        likert_mapping = pd.read_csv(f)
    likert_mapping.merge(likert_questions, on=['question_id_details', 'question_id_enum', 'question_id', 'question_id_short', 'text_answer', 'value'], how='right')

    filepath = os.path.join(mapping_path, f'{likert_mapping_name}{today_str}.csv')
    with fsspec.open(filepath, mode='w') as f:
        likert_mapping.to_csv(f)

    return

'''
Run Function
'''

df = transform_qualtrics()
mapped = write_mappings(df)



'''
Write lines
'''



# if __name__ == '__main__':
#     ctx = orient_snowflake()
#     df = transform_qualtrics(ctx)
#     write_to_snowflake(ctx, df)
#     write_mappings(df)
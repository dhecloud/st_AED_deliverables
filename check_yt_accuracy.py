from sklearn.metrics import f1_score, confusion_matrix, classification_report
import os
import srt
import re

CLASSES = {'gunshot_gunfire':0, 'speech':1, 'crowd_scream':2, 'explosion':3, 'breaking':4, 'siren':5, 'motor_vehicle_road':6, 'crying_sobbing':7, 'others':8, 'silence': 9}
answers_folder = 'youtube_test_answers'
generated_subs_folder = 'youtube_test_set'
paths = [p for p in os.listdir(generated_subs_folder) if p.endswith('.srt')]

def extract_classes_from_srt(srt_line, k = -1):
    re_pattern = r'\d+'
    cleaned_content =  srt_line.content.strip()[4:] #remove model prefix and whitespaces
    cleaned_content=re.sub(re_pattern,'', cleaned_content)
    cleaned_content = cleaned_content.replace('crowd/screaming','crowd_scream') #fix old names
    cleaned_content = cleaned_content.replace('crowd_screaming','crowd_scream')
    cleaned_content = cleaned_content.replace('gunfire_gunshot','gunshot_gunfire') 
    cleaned_content = [x.strip() for x in cleaned_content.split(':')][1:]
    cleaned_content = [x for x in cleaned_content if x is not ''] #remove blanks
    print(cleaned_content)
    assert all(x in CLASSES.keys() for x in cleaned_content)
    return cleaned_content

correct=0
wrong=0
others = 0
silence = 0

ans_idxs = []
cand_idxs = []


for p in paths:
    answer_file = p.replace('.srt', '_answer.srt')
    answer_file_path = os.path.join(answers_folder, answer_file)
    candidate_file_path = os.path.join(generated_subs_folder, p)
    assert os.path.isfile(answer_file_path)
    assert os.path.isfile(candidate_file_path)

    with open(candidate_file_path,'r') as f:
        lines = f.readlines() 
        lines = '\n'.join(lines)
        # print(lines)
        candidate_srts = list(srt.parse(lines))

    with open(answer_file_path,'r') as f:
        lines = f.readlines() 
        lines = '\n'.join(lines)
        # print(lines)
        answer_srts = list(srt.parse(lines))

    for i, sub in enumerate(zip(answer_srts,candidate_srts)):
        answer_matched_substrings = extract_classes_from_srt(sub[0])
        candidate_matched_substrings = extract_classes_from_srt(sub[1])
        ans_idxs.append(CLASSES[answer_matched_substrings[0]])
        cand_idxs.append(CLASSES[candidate_matched_substrings[0]])
        if answer_matched_substrings[0] == 'others':
            others += 1
        if answer_matched_substrings[0] == 'silence':
            silence += 1
        elif answer_matched_substrings[0] in candidate_matched_substrings:
            correct +=1
        else:
            wrong += 1

print('='*100)
print(f'accuracy including others: {correct/(correct+wrong+others)}')
print(f"micro f1_score including others and silence: {f1_score(ans_idxs, cand_idxs, average = 'micro')}")
print(f"macro f1_score including others and silence: {f1_score(ans_idxs, cand_idxs, average = 'macro')}")
print(classification_report(ans_idxs, cand_idxs))

pop_idxes = [i for i, x in enumerate(ans_idxs) if x in [8,9]]
cand_idxs = [x for i,x in enumerate(cand_idxs) if i not in pop_idxes]
ans_idxs = [x for i,x in enumerate(ans_idxs) if i not in pop_idxes]

print('='*100)
print(f'accuracy excluding others: {correct/(correct+wrong)}')
print(f"micro f1_score excluding others and silence: {f1_score(ans_idxs, cand_idxs, average = 'micro')}")
print(f"macro f1_score excluding others and silence: {f1_score(ans_idxs, cand_idxs, average = 'macro')}")
print(CLASSES)
print(classification_report(ans_idxs, cand_idxs))
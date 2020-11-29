
def read_label_desc(file_name):
    fm = open(file=file_name,mode='rt')
    res = []
    for sub in fm:
        a = sub.split(',')
        res.append(a[0])
    fm.close()
    return res

def read_train_subjects(file_name):
    fm = open(file=file_name,mode='rt')
    res = []
    for sub in fm:
        res.append(str(sub.strip('\n')))
    fm.close()
    return res


def read_config_ini(file_name):
    fm = open(file=file_name,mode='rt')
    res = {}
    for sub in fm:
        res.append(str(sub.strip('\n')))
    fm.close()
    return res
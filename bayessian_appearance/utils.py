
import configparser


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
        line= str(sub.strip('\n'))
        line=line.split(',')
        res.update({line[0]:line[1]})
    fm.close()
    return res

def read_modalities_config(file_name):
    cfg = configparser.ConfigParser()
    cfg.read(file_name)
    res = [ ]
    keys = [x for x in cfg['modalities']]
    for i in range(len(keys)):
        res.append( [keys[i],cfg['modalities'][keys[i]]])
    return res




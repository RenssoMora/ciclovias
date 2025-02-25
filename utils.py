import sys
import os
import argparse
import json
import re
import yaml
import random 
import time
import numpy as np
from sys import platform

################################################################################
################################################################################
def u_detect_environment():
    in_colab    = 'google.colab' in sys.modules
    system_os   = "Windows" if os.name == 'nt' else 'Unix'
    return in_colab, system_os

################################################################################
################################################################################
def u_loadJson(file_name):
    with open(file_name) as f:
        data = json.load(f)
    return data

################################################################################
################################################################################
def u_fileList2array(file_name):
    print('Loading data from: ' + file_name)
    F = open(file_name,'r') 
    lst = []
    for item in F:
        item = item.replace('\\', '/').rstrip()
        lst.append(item)
    F.close()
    return lst

################################################################################
################################################################################
def u_fileList2array_(file_name):
    '''
    It loads data from file list which has the first row as the main root
    '''
    print('Loading data from: ' + file_name)
    F       = open(file_name,'r') 
    root    = F.readline().strip() 
    lst = []
    for item in F:
        #item = item.replace('\\', '/').rstrip()
        lst.append(item)
    F.close()
    return root, lst

################################################################################
################################################################################
def u_save2File(file_name, data):
    print('Saving data in: ' + file_name)
    F = open(file_name,'w') 
    F.write(data)
    F.close()

################################################################################
################################################################################
def u_saveList2File(file_name, data):
    print('Saving data in: ' + file_name)
    F = open(file_name,'w') 
    for item in data:
        item = item.strip()
        F.write(item + '\n')
    F.close()

################################################################################
################################################################################
def u_fileNumberList2array(file_name):
    print('Loading data from: ' + file_name)
    F = open(file_name,'r') 
    lst = []
    for item in F:
        if len(item) > 0:
            lst.append(float(item))
    F.close()
    return lst

################################################################################
################################################################################
def u_fileNumberMat2array(file_name):
    print('Loading data from: ' + file_name)
    F = open(file_name,'r') 
    lst = []
    for item in F:
        if len(item) > 0:
            item = item.split(' ')
            sub  = []
            for i in item:
                sub.append(float(i))
            lst.append(sub)
    F.close()
    return lst

################################################################################
################################################################################
''' save string matrix estructure into file'''
def u_fileString2DMat2array(file_name, token):
    print('Loading data from: ' + file_name)
    F = open(file_name,'r') 
    lst = []
    for item in F:
        if len(item) > 0:
            item = item.split(token)
            lst.append(tuple(item))
    F.close()
    return lst

################################################################################
################################################################################
def u_saveArray2File(file_name, data):
    print('Saving data in: ' + file_name)
    F = open(file_name,'w') 
    for item in data:
        F.write(str(item))
        F.write('\n')
    F.close()

################################################################################
################################################################################
def u_saveFlist2File(file_name, root, data):
    print('Saving data in: ' + file_name)
    F = open(file_name,'w') 
    F.write(root)
    for item in data:
        F.write('\n')
        F.write(str(item))
        
    F.close()

################################################################################
################################################################################
def u_saveArrayTuple2File(file_name, data):
    print('Saving data in: ' + file_name)
    F = open(file_name,'w') 
    for item in data:
        line = ''
        for tup in item:
            line += str(tup) + ' '
        F.write(line.strip())
        F.write('\n')
    F.close()
################################################################################
################################################################################
'''
Save dict into file, recommendably [.json]
'''
def u_saveDict2File(file_name, data):
    print ('Saving Dict data in: ', file_name)
    with open(file_name, 'w') as outfile:  
        json.dump(data, outfile)

################################################################################
################################################################################
def u_mkdir(directory):
    '''Crea un directorio si no existe'''
    if not os.path.exists(directory):
        print('Directorio creado en ', directory)
        os.makedirs(directory)

################################################################################
################################################################################
'''
it returns the complete file list in a list
'''
def u_listFileAll(directory, token):
    list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(token):
                list.append(root +'/'+ file)
    
    return sorted (list, key = u_stringSplitByNumbers)
    
################################################################################
################################################################################
'''
it returns a vector with separate root and files
'''
def u_listFileAllVec(directory, token, wdir = True):
    list = []
    for root, dirs, files in os.walk(directory):
        root        = root.replace('\\', '/') 
        sub_list    =[root.replace(directory, ''),[]]
        for file in files:
            if file.endswith(token) or file.endswith(token.upper()):
                if wdir:
                    file = root + '/' + file
                sub_list[1].append(file)
        if len(files) > 0:
            sub_list[1] = sorted (sub_list[1], key = u_stringSplitByNumbers)
            list.append(sub_list)

    return list

################################################################################
################################################################################
def u_listFileAllDic(directory, token=None):
    '''it returns a dictionary with complete map
    '''
    out     = {}
    files   = []
    if token != None:
        for entry in os.listdir(directory):
            local = os.path.join(directory, entry)
            if os.path.isdir(local):
                out[entry] = u_listFileAllDic(local, token)
            else:
                if entry.endswith(token) or \
                   entry.endswith(token.upper()):
                    files.append(entry)
    else:
        for entry in os.listdir(directory):
            local = os.path.join(directory, entry)
            if os.path.isdir(local):
                out[entry] = u_listFileAllDic(local, token)
            else:
                files.append(entry)
    
    if len(files):
        files   = sorted (files, key = u_stringSplitByNumbers)
        out['_files'] = files
    
    out['_path'] = directory

    return out

################################################################################
################################################################################
def u_listFileAllDic_2(base, base_local, token=None):
    '''it returns a dictionary with complete map, the path is separated
    '''
    out     = {}
    files   = []
    path    = base + base_local
    if token != None:
        for entry in os.listdir(path):
            local = u_joinPath([base + base_local, entry])
            if os.path.isdir(local):
                out[entry] = u_listFileAllDic_2(base, base_local+'/'+entry, token)
            else:
                if entry.endswith(token) or \
                   entry.endswith(token.upper()):
                    files.append(entry)
    else:
        for entry in os.listdir(path):
            local = u_joinPath([base + base_local, entry])
            if os.path.isdir(local):
                out[entry] = u_listFileAllDic_2(base, base_local+'/'+entry, token)
            else:
                files.append(entry)
    
    if len(files):
        files   = sorted (files, key = u_stringSplitByNumbers)
        out['_files'] = files
    
    out['_path'] = base_local

    return out


################################################################################
################################################################################
def u_getPath(file):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('inputpath', nargs='?', 
                        help='The input path. Default = conf.json')
    args = parser.parse_args()
    return args.inputpath if args.inputpath is not None else file

################################################################################
################################################################################
def u_loadFileManager(directive, token = ''):
    print(directive)
    if os.path.isfile(directive):
        file_list = []
        file = open(directive)
        for item in file:
            file_list.append(item)
    else:
        file_list   = u_listFileAll(directive, token)

    return sorted(file_list, key = u_stringSplitByNumbers)

################################################################################
################################################################################
''' console bar animation of process'''
def u_progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '>' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush() 
    
################################################################################
################################################################################
''' init a list with different list'''
def u_init_list_of_objects(size):
    list_of_objects = list()
    for i in range(0,size):
        list_of_objects.append( list() ) #different object reference each time
    return list_of_objects

################################################################################
################################################################################
def u_replaceStrList(str_list, token1, token2):
    ''' replace string in a list of strings'''
    for i in range(len(str_list)):
        str_list = str_list.replace(token1, token2)
    return str_list

################################################################################
################################################################################
''' split string by alfanumerical'''
def u_stringSplitByNumbers(x):
    r = re.compile('(\d+)')
    l = r.split(x)
    return [int(y) if y.isdigit() else y for y in l]

################################################################################
################################################################################
''' join string vector in only one string using a defined token'''
def u_fillVecToken(names, token = ' '):
    ans = names[0]
    for i in range(1, len(names)):
        if names[i] != '':
            ans += token + names[i] 
    return ans

################################################################################
################################################################################
''' os  similar to joinpath of python os'''
def u_joinPath(names):
    return u_fillVecToken(names, '/')

################################################################################
################################################################################
''' change into values for determinate key that contains token _pt '''
def u_look4PtInDict(dict_, root):
       for item in dict_:
           if item.find('_pt') > -1:
            dict_[item] = u_joinPath([root, dict_[item]])

################################################################################
################################################################################
def u_divideList(l, n):
    '''divide list into sublist o size n
    looping till length l
    reference variable
    yield acts like list generator
    use: a = list(u_divideList(l, n))
    '''
    for i in range(0, len(l), n):
        yield l[i:i + n]

################################################################################
################################################################################
def u_sec2dhms(seconds: int) -> tuple:
    '''Converts seconds to day, hours, minutes, seconds
    '''
    (days, remainder)   = divmod(seconds, 86400)
    (hours, remainder)  = divmod(remainder, 3600)
    (minutes, seconds)  = divmod(remainder, 60)
    return (days, hours, minutes, seconds)

################################################################################
################################################################################
################################################################################
def __iterdict_recursive(d, pre, sbase):
    keys = list(d.keys())

    if len(keys) > 0:   
        for k in keys[:-1]:
            
            if str(type(d[k])) == "<class 'dict'>":
                print(sbase + pre[2]+ k)
                __iterdict_recursive(d[k], pre, pre[1] + sbase + pre[0])
                
            else:
                print(sbase + pre[2]+ k +' :  '+ str(type(d[k])))
        if str(type(d[keys[-1]])) == "<class 'dict'>":
            print(sbase + pre[3] + keys[-1])
            __iterdict_recursive(d[keys[-1]], pre, pre[0]+ sbase + pre[0])
            
        else:
            print(sbase + pre[3]+ keys[-1] +' :  ' +str(type(d[keys[-1]])))
 
def u_iterdict(d):
    # prefix components:
    space   = '    '
    branch  = '│   '
    # pointers:
    tee     = '├── '
    last    = '└── '
    
    pre     = [space, branch, tee, last]

    print('data')
    __iterdict_recursive(d, pre, '')

################################################################################
################################################################################
################################################################################
def u_loadYaml(file_name:str):
  '''   It reads a yml file and Returns a class format of dictionary 
  the good is you can format strings in yml employing fstring
  '''
  with open(file_name) as f:
    # use safe_load instead load
    dataMap = yaml.safe_load(f)
    data    = u_dict2class(dataMap)
    return data

################################################################################
################################################################################
################################################################################
def u_saveYaml(file_name:str, data):
    '''   It saves a class format of dictionary into a yml file 
    ''' 
    with open(file_name, 'w') as f:
        yaml.dump(u_class2dict(data), f)
    print('Data saved in: ', file_name)

################################################################################
################################################################################
################################################################################
class _MiClase:
    def __init__(self, kwargs:dict, variables={}):
      # Primero, establecer todos los atributos sin formatear
        for key, value in kwargs.items():
            if isinstance(value, dict):
                setattr(self, key, _MiClase(value, variables))  # Convertir diccionarios anidados en instancias de MiClase
            else:
                setattr(self, key, value)
                variables[key] = value
      
      # Luego, formatear las cadenas que contengan '{}'
        for key, value in self.__dict__.items():
            if isinstance(value, str) and '{' in value and '}' in value:
                res = re.findall(r'\{.*?\}', value)
                try:
                    for k in res:
                        value = value.replace(k, str(variables[k[1:-1]]))
                except KeyError as e:
                  print(f"Advertencia: La clave '{e.args[0]}' no se encontró en el diccionario. La cadena '{key}' no se pudo formatear completamente.")
                setattr(self, key, value)

    def __str__(self, level=0):
        result = []
        indent = '    ' * level  # Usamos cuatro espacios para la indentación
        for key, value in self.__dict__.items():
            if isinstance(value, _MiClase):
                result.append(f"{indent}{key}: \n{value.__str__(level + 1)}")
            else:
                result.append(f"{indent}{key}: {value}")
        return '\n'.join(result)

def u_dict2class(d):
    return _MiClase(d)

################################################################################
################################################################################
################################################################################
def u_class2dict(obj):
    """It converts a class instance into a dictionary recursively
    """
    if isinstance(obj, dict):
        return {k: u_class2dict(v) for k, v in obj.items()}
    elif hasattr(obj, "__dict__"):  # Si es una instancia de clase
        return {k: u_class2dict(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, list):  # Si es una lista, procesar cada elemento
        return [u_class2dict(item) for item in obj]
    else:
        return obj  # Si es un tipo primitivo, devolver tal cual

################################################################################
################################################################################
################################################################################
def u_getLastFile(directory, base_token, flag=True):
    ''' get the last file in a directory
    it starts in 1 and increases until it finds the last file
    it recieves the base_token and the end_token number is between them
    '''
    count = 1 
    while os.path.exists(os.path.join(directory, f"{base_token}_{count}")):
        count += 1    

    if count == 1:
        return f"{base_token}_{count}"
    
    return f"{base_token}_{count if flag else count-1}"
    

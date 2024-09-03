import json

def saveDictToJson(data,filepath):
    with open(filepath,'w',encoding='utf-8') as f:
        json.dump(data,f,ensure_ascii=False)

def readDictFromJson(filepath):
    with open(filepath,'r',encoding='utf-8') as f:
        data = json.load(f)
        return data

def readConfig(filepath="./config.yml"):
    import yaml
    with open(filepath, 'r', encoding='utf-8') as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)
    return result
from collections import defaultdict
from email.policy import default
import torch, pickle, json
from torch.utils.data import Dataset, DataLoader

def get_son2parent(csv_path):
    son2parent = dict()
    son = []
    parent = []
    grand_parent = []
    with open (csv_path,'r') as fp:
        lines = fp.readlines()
        for line in lines:
            if line[-1] == '\n':
                line = line[:-1]
            tmp_list = line.split(',')
            
            for i in range(len(tmp_list) - 1): # ignore last digit.
                key, value = tmp_list[i], tmp_list[i+1]
                son2parent[key] = value
            for i in range(len(tmp_list) - 3): # ignore root.
                key, value = tmp_list[i].replace(" ",""), tmp_list[i+1]
                son.append(key)
                parent.append(value)
            for i in range(len(tmp_list)-3): # ignore root.
                _, value = tmp_list[i].replace(" ",""), tmp_list[i+2]
                grand_parent.append(value)
    if '' in son2parent.keys():
        del son2parent['']
    temp = defaultdict(lambda: len(temp))
    parent_id = [temp[i] for i in parent]
    grand_parent_id = [temp[i] for i in grand_parent]
    
    return son2parent, son, parent, parent_id , grand_parent, grand_parent_id

class My_DS(Dataset):
    def __init__(self, X, y, emb):
        self.X = X
        self.y = y
        self.emb = emb
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        
        x_batch = self.X[index, :]
        y_batch = self.y[index]
        a_batch = self.emb[y_batch,:]
        
        return x_batch, y_batch, a_batch

def get_dataloader(Xtr, Xval, ytr, yval, emb, batch_size):
    
    train_dataset = My_DS(X=Xtr, y=ytr, emb=emb)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    val_dataset = My_DS(X=Xval, y=yval, emb=emb)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    return train_loader, val_loader

def split_act_data(feat, label, fns, anno_fn): # This only for ActivityNet.

    with open(anno_fn) as json_file:
        data = json.load(json_file)

    anno_tr = {key:value for key,value in data['database'].items() if value['annotations']}

    training_inx, validation_inx = [], []

    for feat_fn in fns:
        # Using file name to get the corresponding clip_id, 找到文件名及对应clip_id
        key = feat_fn.split('.')[0].split('_')[-1]
        key = '_'.join(feat_fn.split('.')[0].split('_')[2:])
        clip_inx = feat_fn.split('.')[0].split('_')[0]
        clip_inx = int(clip_inx)

        # Get the video annotation, i.e. the action label.
        video_anno = anno_tr[key]
        if video_anno['subset'] == 'validation':
            training_inx.append(False), validation_inx.append(True)
        elif video_anno['subset'] == 'training':
            training_inx.append(True), validation_inx.append(False)
    
    training_inx, validation_inx = torch.tensor(training_inx), torch.tensor(validation_inx)
    
    Xtr, Xval = feat[training_inx], feat[validation_inx]
    ytr, yval = label[training_inx], label[validation_inx]
    assert Xtr.dim() == 2 and ytr.dim() == 1
    
    return Xtr, Xval, ytr, yval

def get_activitynet_dataset(feat_path = './data.pickle', anno_fn = './activity_net.v1-3.json'):
    # Generate X, the feature and y, the label.
    with open(feat_path,'rb') as f:
        data1 = pickle.load(f)
        
    label_set = list(set(data1['label']))
    label_set.sort() # make sure that the label set is sorted for one-hot encoding.

    label = [label_set.index(item) for item in data1['label']]
    label = torch.tensor(label)
    feat, fns = data1['feat'], data1['fn']
    
    Xtr, Xval, ytr, yval = split_act_data(feat, label, fns, anno_fn)
    
    return Xtr, Xval, ytr, yval, label_set


def get_emb(emb_type, emb_file, label_set):
    
    n_cls = len(label_set)
    if emb_type == 'rand':
        emb = torch.rand(n_cls,300)
    
    elif emb_type == 'wacv':
        with open(emb_file, 'rb') as pickle_file:
            content = pickle.load(pickle_file)
            emb = content['embedding']
            emb = torch.tensor(emb).cuda()
        

    elif emb_type == 'oh':
        one_hot = torch.zeros(n_cls, n_cls).long()
        emb = one_hot.scatter_(dim=1, index=torch.unsqueeze(torch.arange(n_cls), dim=1), src=torch.ones(n_cls, n_cls).long())
        emb = emb.float()

    elif emb_type == 'glove':
        data2 = torch.load(emb_file, map_location='cpu')
        emb_names = data2['objects']
        ext_emb = data2['embeddings'] # n_cls x dim

        emb = torch.zeros(n_cls,ext_emb.shape[1])
        emb = emb.cuda()
        for i in range(n_cls):
            pos = emb_names.index(label_set[i])
            emb[i] = ext_emb[pos,:]

    elif emb_type == 'hyp':
        data2 = torch.load(emb_file, map_location='cpu')
        emb_names = data2['objects']
        
        if "Basetball Pitch" in emb_names:
            emb_names[emb_names.index("Basetball Pitch")] =  "Basketball Pitch"
        if "Basketball"  in emb_names:
            emb_names[emb_names.index("Basketball" )] = "Basketball Shooting"
        if "Billiards" in emb_names:
            emb_names[emb_names.index("Billiards")] = "Billiards Shot"
        if "Mixing" in emb_names:
            emb_names[emb_names.index("Mixing")] = "Mixing Batter"
        if  "Walking with dog" in emb_names:
            emb_names[emb_names.index("Walking with dog")] =  "Walking with a dog"
        if  "Salsa Spin" in emb_names:
            emb_names[emb_names.index("Salsa Spin")] = "Salsa Spins" 
        if "Baby crawling " in emb_names:
            emb_names[emb_names.index("Baby crawling ")] = "Baby crawling"
            
        
        import string
        
        ext_emb = data2['embeddings'] # 272 x dim
        emb = torch.zeros(n_cls,ext_emb.shape[1])
        for i in range(n_cls):
            
            
            pos = emb_names.index(label_set[i])
            emb[i] = ext_emb[pos,:]

    elif emb_type == 'cone':
        data2 = torch.load(emb_file, map_location='cpu')
        if type(data2) is zip:
            data2 = dict(data2)
        emb_names = list(data2.keys())
        ext_emb = list(data2.values()) # 271 x dim, root is discarded 
        ext_emb = torch.tensor(ext_emb)

        emb = torch.zeros(n_cls,ext_emb.shape[1])
        for i in range(n_cls):
            pos = emb_names.index(label_set[i])
            emb[i] = ext_emb[pos,:]
    
    return emb

def get_kineticslike_dataset(train_pth_path, valid_pth_path):
    data_train = torch.load(train_pth_path)
    data_val = torch.load(valid_pth_path)

    label_set = list(set(data_train['label']))
    label_set.sort() # 有Sort很重要
    
    Xtr, Xval = data_train['feat'], data_val['feat']
    ytr = torch.tensor([label_set.index(item )for item in data_train['label']])
    yval = torch.tensor([label_set.index(item )for item in data_val['label']])
    
    return Xtr, Xval, ytr, yval, label_set

def get_ucf_dataset(train_pth_path, valid_pth_path):
    data_train = torch.load(train_pth_path)
    data_val = torch.load(valid_pth_path)
    
    
    class_names = ["Apply Eye Makeup", "Apply Lipstick","Blow Dry Hair","Brushing Teeth"
    ,"Cutting In Kitchen"
    ,"Hammering"
    ,"Hula Hoop"
    ,"Juggling Balls"
    ,"Jump Rope"
    ,"Knitting"
    ,"Mixing"
    ,"Mopping Floor"
    ,"Nun chucks"
    ,"Pizza Tossing"
    ,"Shaving Beard"
    ,"Skate Boarding"
    ,"Soccer Juggling"
    ,"Typing"
    ,"Writing On Board"
    ,"Yo Yo"
    ,"Baby crawling "
    ,"Blowing Candles"
    ,"Body Weight Squats"
    ,"Handstand Pushups"
    ,"Handstand Walking"
    ,"Jumping Jack"
    ,"Lunges"
    ,"Pull Ups"
    ,"Push Ups"
    ,"Rock Climbing Indoor"
    ,"Rope Climbing"
    ,"Swing"
    ,"Tai Chi"
    ,"Trampoline Jumping"
    ,"Walking with dog"
    ,"Wall Pushups"
    ,"Band Marching"
    ,"Haircut"
    ,"Head Massage"
    ,"Military Parade"
    ,"Salsa Spin"
    ,"Drumming"
    ,"Playing Guitar"
    ,"Playing Piano"
    ,"Playing Tabla"
    ,"Playing Violin"
    ,"Playing Cello"
    ,"Playing Daf"
    ,"Playing Dhol"
    ,"Playing Flute"
    ,"Playing Sitar"
    ,"Archery"
    ,"Balance Beam"
    ,"Baseball Pitch"
    ,"Basketball" 
    ,"Basketball Dunk"
    ,"Bench Press"
    ,"Biking"
    ,"Billiards"
    ,"Bowling"
    ,"Boxing Punching Bag"
    ,"Boxing Speed Bag"
    ,"Breaststroke"
    ,"Clean and Jerk"
    ,"Cliff Diving"
    ,"Cricket Bowling"
    ,"Cricket Shot"
    ,"Diving"
    ,"Fencing"
    ,"Field Hockey Penalty"
    ,"Floor Gymnastics"
    ,"Frisbee Catch"
    ,"Front Crawl"
    ,"Golf Swing"
    ,"Hammer Throw"
    ,"High Jump"
    ,"Horse Race"
    ,"Horse Riding"
    ,"Ice Dancing"
    ,"Javelin Throw"
    ,"Kayaking"
    ,"Long Jump"
    ,"Parallel Bars"
    ,"Pole Vault"
    ,"Pommel Horse"
    ,"Punch"
    ,"Rafting"
    ,"Rowing"
    ,"Shotput"
    ,"Skiing"
    ,"Skijet"
    ,"Sky Diving"
    ,"Soccer Penalty"
    ,"Still Rings"
    ,"Sumo Wrestling"
    ,"Surfing"
    ,"Table Tennis Shot"
    ,"Tennis Swing"
    ,"Throw Discus"
    ,"Uneven Bars"
    ,"Volleyball Spiking"]  
    
    label_set = list(set(class_names))
    label_set.sort() # 有Sort很重要
    
    Xtr, Xval = data_train['features'], data_val['features']
    ytr = data_train['labels']
    yval = data_val['labels']
    
    return Xtr, Xval, ytr, yval, label_set



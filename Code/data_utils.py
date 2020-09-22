import glob
from natsort import natsorted
import os

# Mappings from dataset parameters to filename code
s_dict = {'table_start': 0, 'hand_start': 1, 'off_start': 2}
fi_dict = {'nothing': 0, 'pasta': 1, 'rice': 2, 'water': 3}
fu_dict = {'zero': 0, 'fifty': 1, 'ninety': 2}
b_dict = {'regular': 0, 'textured': 1}
l_dict = {'light0': 0, 'light1': 1}
c_dict = {'c1': 1, 'c2': 2, 'c3': 3, 'c4': 4}
obj_id_dict = {1: 'red cup', 2: 'small white cup',3:'small transparent cup',4:'green glass',5:'wine glass',
              6:'champagne flute glass', 7:'cereal box',8:'biscuit box',9:'tea box'} 

valid_dict = {'s': list(s_dict.keys()), 
              'fi': list(fi_dict.keys()),
              'fu': list(fu_dict.keys()),
              'b': list(b_dict.keys()),
              'l': list(l_dict.keys()),
              'c': list(c_dict.keys()),
              'obj_id': list(obj_id_dict.keys()),
             }

def retrieve_data(root_path, obj_id, s, fi, fu, b, l, c=[]):
    ''' Given an object ID and condition parameters, returns the filenames of each
        modality present in the dataset.

    '''

    if ((fi == 'nothing' and (fu =='fifty' or fu =='ninety')) or (fi == 'pasta' and fu == 'zero') or (fi == 'rice' and fu=='zero') or (fi=='water' and fu=='zero')): 
        #print('error')
        return -1
    for i in range(1,len(c),1):
        if c[i] not in valid_dict['c']:
            return -1
    if  (obj_id not in obj_id_dict) or (s not in valid_dict['s']) or (fi not in valid_dict['fi']) or (fu not in valid_dict['fu']) or (b not in valid_dict['b']) or (l not in valid_dict['l']) :
        return -1

    _obj_id = obj_id
    _s_id = s_dict[s]
    _fi_id = fi_dict[fi]
    _fu_id = fu_dict[fu]
    _b_id = b_dict[b]
    _l_id = l_dict[l]
    _c_id = []

    for i in range(0,len(c),1):
        _c_id.append(c_dict[c[i]])
    if(len(c)==0):
        _c_id = [1,2,3,4]

    input_string = 's'+str(_s_id)+'_fi'+str(_fi_id)+'_fu'+str(_fu_id)+'_b'+str(_b_id)+'_l'+str(_l_id)

    audio_path = os.path.join(root_path, str(_obj_id) + "/audio/" + input_string + "*")
    audio_list = glob.glob(audio_path)[0]

    calib_list = []
    for i in range(0,len(_c_id),1):
        calib_path = os.path.join(root_path, str(_obj_id) + "/calib/" + input_string+'_c' + str(_c_id[i]) + '*')
        calib_list.append(glob.glob(calib_path)[0])

    depth_list = []
    for i in range(0,len(_c_id),1):
        depth_path = os.path.join(root_path, str(_obj_id) + "/depth/" + input_string + '/c' + str(_c_id[i]) + '/*')
        depth_list.append(natsorted(glob.glob(depth_path), key=lambda y: y.lower()))

    imu_path = os.path.join(root_path, str(_obj_id) + "/imu/" + input_string + "*")
    imu_list = tuple(glob.glob(imu_path))


    ir_list=[]
    for i in range(0,len(_c_id),1):
        ir_path = os.path.join(root_path, str(_obj_id) + "/ir/" + input_string + '_c' + str(_c_id[i]) + '*')
        ir_list.append(glob.glob(ir_path))


    rgb_list = []
    for i in range(0,len(_c_id),1):
        rgb_path = os.path.join(root_path, str(_obj_id) + "/rgb/" + input_string + '_c' + str(_c_id[i]) + '*')
        rgb_list.append(glob.glob(rgb_path)[0])
    
    
    output_dict = {'audio': audio_list,'calib':calib_list,'depth':depth_list,'imu':imu_list,'ir':ir_list,'rgb':rgb_list}
    return output_dict

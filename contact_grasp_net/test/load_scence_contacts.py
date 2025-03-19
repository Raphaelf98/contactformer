import numpy as np
import os
import glob
acronym_dir = "/home/raphael/thesis/contact_graspnet_pytorch/acronym"
scene_contacts_path = 'scene_contacts'
valid_contact_infos = []
scene_contact_paths = sorted(glob.glob(os.path.join(acronym_dir, scene_contacts_path, '*')))

for contact_path in scene_contact_paths:

    contact_id = contact_path.split('/')[-1].split('.')[0]
        
    try:
        npz = np.load(contact_path, allow_pickle=False)
        contact_info = {'scene_contact_id': contact_id,
                            'scene_contact_points':npz['scene_contact_points'],
                            'obj_paths':npz['obj_paths'],
                            'obj_transforms':npz['obj_transforms'],
                            'obj_scales':npz['obj_scales'],
                            'grasp_transforms':npz['grasp_transforms']}
        valid_contact_infos.append(contact_info)
    except:

        print(f'corrupt scene: {contact_path}')
print(len(valid_contact_infos))
print(valid_contact_infos[0]['scene_contact_points'].shape)

print(valid_contact_infos[0]['obj_paths'])
print(valid_contact_infos[0]['obj_transforms'].shape)
print(valid_contact_infos[0]['obj_scales'].shape)
print(valid_contact_infos[0]['grasp_transforms'].shape)
